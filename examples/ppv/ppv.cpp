// Position-parallel verification (PPV) driver and baselines, greedy only.
//
//   ar : plain decode of the target                                   (1 process)
//   sd : speculative decoding, drafter + batched verify of g tokens   (1 process)
//   ppv: speculative decoding where the g = world tokens of each verify are computed one per rank
//        (LLAMA_PPV_PEERS / LLAMA_PPV_RANK). Rank 0 runs the drafter itself, or, with
//        --drafter host:port, receives the drafts from a separate drafter process.
//   drafter: the drafter on its own board (--listen port, -g = verifier world). After sending a block
//        it keeps drafting ahead, assuming the block is fully accepted and guessing the bonus token;
//        when the verify result confirms the guess, the next block is sent without drafting delay.
//        --branches N: also pre-draft up to N likely miss outcomes in forked KV sequences (see below).
//
// usage: llama-ppv -m target.gguf [-md draft.gguf] -f prompt.txt [-n 256] [--mode ar|sd|ppv|drafter] [-g 4] [-t 4] [-c 4096]
//                  [--drafter host:port] [--listen port] [--branches N] [--group G] [--replica d:D]
//                  (--drafter takes a comma-separated list of drafter replicas)
// The prompt file is used verbatim (special tokens parsed), so apply the chat template beforehand.

#include "llama.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

static llama_token argmax(const float * logits, int n_vocab) {
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

static void send_all(int fd, const void * buf, size_t n) {
    const char * p = (const char *) buf;
    while (n > 0) { ssize_t w = send(fd, p, n, 0); if (w <= 0) { fprintf(stderr, "send failed\n"); exit(1); } p += w; n -= (size_t) w; }
}
static void recv_all(int fd, void * buf, size_t n) {
    char * p = (char *) buf;
    while (n > 0) { ssize_t r = recv(fd, p, n, 0); if (r <= 0) { fprintf(stderr, "recv failed\n"); exit(1); } p += r; n -= (size_t) r; }
}
static void nodelay(int fd) { int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)); }

static int listen_accept(int port) {
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1; setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_ANY); a.sin_port = htons(port);
    if (bind(lfd, (sockaddr *) &a, sizeof(a)) != 0 || listen(lfd, 1) != 0) { fprintf(stderr, "bind/listen failed\n"); exit(1); }
    int fd = accept(lfd, nullptr, nullptr);
    close(lfd); nodelay(fd);
    return fd;
}

static int connect_retry(const std::string & hp) {
    const size_t c = hp.rfind(':');
    addrinfo hints = {}, * res = nullptr; hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(hp.substr(0, c).c_str(), hp.substr(c + 1).c_str(), &hints, &res) != 0) { fprintf(stderr, "bad address %s\n", hp.c_str()); exit(1); }
    for (int i = 0; i < 6000; ++i) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (connect(fd, res->ai_addr, res->ai_addrlen) == 0) { freeaddrinfo(res); nodelay(fd); return fd; }
        close(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    fprintf(stderr, "connect to %s failed\n", hp.c_str()); exit(1);
}

// drafter -> verifier rank 0: the block for a round; verifier rank 0 -> every drafter: the block it took
// (type 0, forwarded so replicas that did not draft it can catch up) and the verify result (type 1)
#define PPV_MAX_G 16
struct d2v { int32_t round; int32_t toks[PPV_MAX_G]; };
struct v2d { int32_t type, round, n_acc, bonus, stop; int32_t toks[PPV_MAX_G]; };

struct runner {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    const llama_vocab * vocab = nullptr;
    int n_vocab = 0;
    int n_past  = 0; // tokens whose KV is in the cache

    void load(const std::string & path, int n_ctx, int n_threads, int n_seq = 1) {
        auto mp = llama_model_default_params();
        model = llama_model_load_from_file(path.c_str(), mp);
        if (!model) { fprintf(stderr, "failed to load %s\n", path.c_str()); exit(1); }
        auto cp = llama_context_default_params();
        cp.n_ctx           = n_ctx;
        cp.n_batch         = 512;
        cp.n_ubatch        = 512;
        cp.n_threads       = n_threads;
        cp.n_threads_batch = n_threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.no_perf         = true;
        cp.n_seq_max       = n_seq;
        cp.kv_unified      = true; // branches share the prompt/confirmed prefix cells
        ctx   = llama_init_from_model(model, cp);
        vocab = llama_model_get_vocab(model);
        n_vocab = llama_vocab_n_tokens(vocab);
    }

    // decode toks at positions n_past.. ; logits for the rows in out (or the last row if out is empty)
    void decode(const std::vector<llama_token> & toks, const std::vector<int> & out = {}) {
        for (size_t i0 = 0; i0 < toks.size(); i0 += 512) {
            const int n = (int) std::min<size_t>(512, toks.size() - i0);
            llama_batch b = llama_batch_init(n, 0, 1);
            for (int i = 0; i < n; ++i) {
                b.token[i] = toks[i0 + i];
                b.pos[i] = n_past + i;
                b.n_seq_id[i] = 1;
                b.seq_id[i][0] = 0;
                b.logits[i] = 0;
            }
            b.n_tokens = n;
            if (i0 + n == toks.size()) {
                if (out.empty()) {
                    b.logits[n - 1] = 1;
                } else {
                    for (int r : out) b.logits[r] = 1;
                }
            }
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); exit(1); }
            llama_batch_free(b);
            n_past += n;
        }
    }

    llama_token top(int i) { return argmax(llama_get_logits_ith(ctx, i), n_vocab); }

    void truncate(int n) { // keep positions [0, n)
        llama_memory_seq_rm(llama_get_memory(ctx), 0, n, -1);
        n_past = n;
    }
};

int main(int argc, char ** argv) {
    std::string m_path, md_path, f_path, mode = "ar";
    int n_predict = 256, g = 4, n_threads = 4, n_ctx = 4096, listen_port = 0, n_branch = 0, n_group = 2, rep_id = 0, n_rep = 1;
    std::string drafter_addr;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return std::string(argv[++i]); };
        if      (a == "-m")     m_path    = next();
        else if (a == "-md")    md_path   = next();
        else if (a == "-f")     f_path    = next();
        else if (a == "-n")     n_predict = std::stoi(next());
        else if (a == "-g")     g         = std::stoi(next());
        else if (a == "-t")     n_threads = std::stoi(next());
        else if (a == "-c")     n_ctx     = std::stoi(next());
        else if (a == "--mode") mode      = next();
        else if (a == "--drafter") drafter_addr = next();
        else if (a == "--listen")  listen_port  = std::stoi(next());
        else if (a == "--branches") n_branch    = std::stoi(next());
        else if (a == "--group")    n_group     = std::stoi(next());
        else if (a == "--replica")  { const std::string v = next(); rep_id = std::stoi(v.substr(0, v.find(':'))); n_rep = std::stoi(v.substr(v.find(':') + 1)); }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (m_path.empty() || f_path.empty() || (mode != "ar" && mode != "sd" && mode != "ppv" && mode != "drafter")) {
        fprintf(stderr, "usage: %s -m target.gguf [-md draft.gguf] -f prompt.txt [-n N] [--mode ar|sd|ppv] [-g G] [-t T] [-c C]\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_log_set([](ggml_log_level lvl, const char * txt, void *) { if (lvl >= GGML_LOG_LEVEL_WARN) fputs(txt, stderr); }, nullptr);

    if (mode == "drafter" && n_branch > 0) {
        // ---- drafter on its own board with branch pre-drafting (--branches N), optionally one of D replicas
        // (--replica d:D) on separate boards. After block b_1..b_gam (trunk, seq 0) is known, the candidate
        // verify outcomes (k accepted, bonus) are ranked by the drafter's probabilities: k < gam with bonus =
        // our 2nd..K-th choice at position k+1, and k = gam with bonus = our 1st..K-th choice after b_gam.
        // Candidate i belongs to replica i % D; each replica drafts its first N in forked KV sequences, the top
        // --group G per batched decode. A receiver thread sends a finished matching branch's block as soon as the
        // result lands; every other replica re-drafts from the bonus (same greedy tokens) until a block for the
        // round is forwarded, so the first block in wins at the verifier.
        // The verifier forwards the block it took to all replicas; a replica that did not draft it catches up
        // with one batched decode.
        const int gam = g - 1;
        if (gam > PPV_MAX_G) { fprintf(stderr, "g too large\n"); return 1; }
        const int K = 3;
        runner dft;
        dft.load(md_path.empty() ? m_path : md_path, n_ctx, n_threads, n_branch + 1);
        llama_memory_t mem = llama_get_memory(dft.ctx);
        std::stringstream ss; ss << std::ifstream(f_path).rdbuf();
        const std::string prompt = ss.str();
        std::vector<llama_token> ptoks(prompt.size() + 16);
        const int n_prompt = llama_tokenize(dft.vocab, prompt.c_str(), (int) prompt.size(), ptoks.data(), (int) ptoks.size(), true, true);
        ptoks.resize(n_prompt);
        dft.decode(std::vector<llama_token>(ptoks.begin(), ptoks.end() - 1)); // seq 0, positions [0, n_prompt-1)
        const int fd = listen_accept(listen_port);

        struct pred { llama_token t[K] = { -1, -1, -1 }; float p[K] = { 0, 0, 0 }; };
        struct branch {
            int seq = 0, k = 0, base = 0;  // toks[i] sits at position base + i; toks[0] is the (guessed) bonus
            std::vector<llama_token> toks; // toks[1..] are drafts
            std::vector<pred> pr;          // pr[i]: the prediction that produced toks[i] (i >= 1)
            int n_kv = 0;                  // toks[0, n_kv) are in the KV of `seq`
            double w = 0;
            int choice = 0;                // which of the K choices the bonus guess is
            bool pending = false;          // k = gam guess that needs the trunk's prediction after b_gam
        };
        const int n_vocab = dft.n_vocab;
        auto topk = [&](const float * l) {
            pred r; int id[K] = { -1, -1, -1 };
            for (int i = 0; i < n_vocab; ++i) {
                for (int j = 0; j < K; ++j) {
                    if (id[j] < 0 || l[i] > l[id[j]]) { for (int q = K - 1; q > j; --q) id[q] = id[q - 1]; id[j] = i; break; }
                }
            }
            double z = 0; for (int i = 0; i < n_vocab; ++i) z += std::exp((double) (l[i] - l[id[0]]));
            for (int j = 0; j < K; ++j) { r.t[j] = id[j]; r.p[j] = (float) (std::exp((double) (l[id[j]] - l[id[0]])) / z); }
            return r;
        };

        branch T; T.seq = 0; T.base = n_prompt - 1; T.toks = { ptoks.back() }; T.pr = { pred() };
        std::vector<branch> B;
        int n_rounds = 0, n_match = 0, n_partial = 0, n_early = 0, n_sent = 0, n_catchup = 0;
        long long n_branch_steps = 0, n_branch_tok = 0;

        std::mutex mu; std::condition_variable cv;
        bool has_res = false, has_fwd = false, armed = false;
        int sent_for = -1; // last round this replica sent a block for
        v2d res = {}, fwd = {};
        auto match = [&](const v2d & r) -> branch * {
            for (auto & c : B) if (!c.pending && c.k == r.n_acc && c.toks[0] == r.bonus) return &c;
            return nullptr;
        };
        auto send_block = [&](int round, const branch & x) {
            d2v m = {}; m.round = round;
            for (int i = 0; i < gam; ++i) m.toks[i] = x.toks[1 + i];
            send_all(fd, &m, sizeof(m)); sent_for = round; n_sent++;
        };
        std::thread rx([&]() {
            while (true) {
                v2d m; recv_all(fd, &m, sizeof(m));
                std::lock_guard<std::mutex> lk(mu);
                if (m.type == 0) { fwd = m; has_fwd = true; cv.notify_one(); continue; }
                if (!m.stop && armed) {
                    branch * b = match(m);
                    if (b && (int) b->toks.size() >= gam + 1) { send_block(m.round + 1, *b); n_early++; }
                }
                armed = false; res = m; has_res = true;
                cv.notify_one();
                if (m.stop) break;
            }
        });

        std::unique_lock<std::mutex> lk(mu);
        // one batched decode of toks[n_kv] of each listed branch, without the lock (only this thread appends to
        // the branches); a prediction is appended when the decoded token was the branch's last
        auto step_unlocked = [&](const std::vector<branch *> & bs) {
            lk.unlock();
            llama_batch b = llama_batch_init((int) bs.size(), 0, 1);
            for (size_t i = 0; i < bs.size(); ++i) {
                branch * x = bs[i];
                b.token[i] = x->toks[x->n_kv]; b.pos[i] = x->base + x->n_kv;
                b.n_seq_id[i] = 1; b.seq_id[i][0] = x->seq; b.logits[i] = 1;
            }
            b.n_tokens = (int) bs.size();
            if (llama_decode(dft.ctx, b) != 0) { fprintf(stderr, "decode failed\n"); exit(1); }
            llama_batch_free(b);
            std::vector<pred> ps(bs.size());
            for (size_t i = 0; i < bs.size(); ++i) {
                if (bs[i]->n_kv + 1 == (int) bs[i]->toks.size()) ps[i] = topk(llama_get_logits_ith(dft.ctx, (int) i));
            }
            lk.lock();
            for (size_t i = 0; i < bs.size(); ++i) {
                branch * x = bs[i];
                x->n_kv++;
                if (x->n_kv == (int) x->toks.size()) { x->toks.push_back(ps[i].t[0]); x->pr.push_back(ps[i]); }
            }
        };

        int round = 0;
        while (true) {
            // ---- phase A: the block for `round`
            if (sent_for != round) { // re-draft (or finish a partial branch) until a block for this round is forwarded
                while ((int) T.toks.size() < gam + 1 && !(has_fwd && fwd.round == round)) step_unlocked({ &T });
                if (!(has_fwd && fwd.round == round)) send_block(round, T);
            }
            cv.wait(lk, [&]() { return has_fwd && fwd.round == round; });
            has_fwd = false;
            bool same = (int) T.toks.size() >= gam + 1;
            for (int i = 0; same && i < gam; ++i) same = T.toks[1 + i] == fwd.toks[i];
            if (!same) { // catch up: trunk = toks[0] + forwarded block, decoded in one batch with all predictions
                n_catchup++;
                const int keep0 = std::min(T.n_kv, 1);
                llama_memory_seq_rm(mem, 0, T.base + keep0, -1);
                branch nt; nt.seq = 0; nt.base = T.base; nt.toks = { T.toks[0] }; nt.pr = { pred() };
                if (keep0 == 1) nt.pr.push_back(T.pr.size() > 1 ? T.pr[1] : pred());
                for (int i = 0; i < gam; ++i) nt.toks.push_back(fwd.toks[i]);
                const int n = gam + 1 - keep0;
                lk.unlock();
                llama_batch b = llama_batch_init(n, 0, 1);
                for (int i = 0; i < n; ++i) {
                    b.token[i] = nt.toks[keep0 + i]; b.pos[i] = nt.base + keep0 + i;
                    b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
                }
                b.n_tokens = n;
                if (llama_decode(dft.ctx, b) != 0) { fprintf(stderr, "decode failed\n"); exit(1); }
                llama_batch_free(b);
                std::vector<pred> ps(n);
                for (int i = 0; i < n; ++i) ps[i] = topk(llama_get_logits_ith(dft.ctx, i));
                lk.lock();
                nt.pr.resize(keep0 + 1);
                for (int i = 0; i < n; ++i) nt.pr.push_back(ps[i]);    // pr[keep0 + 1 + i]
                nt.toks.push_back(ps[n - 1].t[0]);                     // our guess after b_gam
                nt.n_kv = gam + 1;
                T = nt;
            }

            // ---- phase B: plan (assignment uses provisional k = gam weights so every replica agrees)
            B.clear();
            {
                std::vector<branch> cand;
                const float prov[K] = { 0.8f, 0.1f, 0.03f };
                double reach = 1.0;
                for (int k = 0; k <= gam; ++k) {
                    for (int c = (k < gam ? 1 : 0); c < K; ++c) {
                        branch x; x.k = k; x.base = T.base + k + 1; x.pr = { pred() }; x.choice = c;
                        if (k < gam) { x.toks = { T.pr[k + 1].t[c] }; x.w = reach * T.pr[k + 1].p[c]; }
                        else         { x.pending = true; x.w = reach * prov[c]; }
                        cand.push_back(x);
                    }
                    if (k < gam) reach *= T.pr[k + 1].p[0];
                }
                std::stable_sort(cand.begin(), cand.end(), [](const branch & a, const branch & b) { return a.w > b.w; });
                int s = 1;
                for (int i = 0; i < (int) cand.size() && i / n_rep < n_branch; ++i) {
                    if (i % n_rep != rep_id) continue;
                    branch x = cand[i]; x.seq = s++;
                    llama_memory_seq_rm(mem, x.seq, -1, -1);
                    B.push_back(x);
                }
            }
            auto resolve = [&]() { // fill k = gam guesses once the trunk has its prediction after b_gam
                const pred & p = T.pr[gam + 1];
                double reach = 1.0; for (int j = 1; j <= gam; ++j) reach *= T.pr[j].p[0];
                for (auto & c : B) if (c.pending) { c.pending = false; c.toks = { p.t[c.choice] }; c.w = reach * p.p[c.choice]; }
                std::stable_sort(B.begin(), B.end(), [](const branch & a, const branch & b) { return a.w > b.w; });
            };
            if ((int) T.toks.size() >= gam + 2) resolve();
            for (auto & c : B) if (!c.pending) llama_memory_seq_cp(mem, 0, c.seq, 0, c.base);
            armed = !has_res;
            while (!has_res) {
                const bool trunk = (int) T.toks.size() < gam + 2;
                std::vector<branch *> act;
                if (trunk) act.push_back(&T);
                for (auto & c : B) {
                    if ((int) act.size() >= n_group) break;
                    if (!c.pending && (int) c.toks.size() < gam + 1) act.push_back(&c);
                }
                if (act.empty()) { cv.wait(lk, [&]() { return has_res; }); break; }
                step_unlocked(act); n_branch_steps++; n_branch_tok += (long long) act.size();
                if (trunk && (int) T.toks.size() >= gam + 2) {
                    resolve();
                    for (auto & c : B) if (c.n_kv == 0 && c.toks.size() == 1) llama_memory_seq_cp(mem, 0, c.seq, 0, c.base);
                }
            }
            // ---- phase C: the result for `round`
            const v2d r = res; has_res = false;
            n_rounds++;
            if (r.stop) break;
            const int k = r.n_acc;
            branch * m = match(r);
            if (m) {
                n_match++;
                if ((int) m->toks.size() < gam + 1) n_partial++;
                llama_memory_seq_rm(mem, 0, m->base, -1);
                llama_memory_seq_cp(mem, m->seq, 0, m->base, -1);
                branch nt = *m; nt.seq = 0;
                T = nt;
            } else {
                // keep trunk toks[0..k] (b_1..b_k accepted); decode any of them not yet in the KV
                const int keep = std::min(T.n_kv, k + 1);
                llama_memory_seq_rm(mem, 0, T.base + keep, -1);
                if (keep < k + 1) {
                    std::vector<llama_token> cu(T.toks.begin() + keep, T.toks.begin() + k + 1);
                    lk.unlock();
                    dft.n_past = T.base + keep;
                    dft.decode(cu);
                    lk.lock();
                }
                branch nt; nt.seq = 0; nt.base = T.base + k + 1; nt.toks = { (llama_token) r.bonus }; nt.pr = { pred() };
                T = nt;
            }
            for (int s = 1; s <= n_branch; ++s) llama_memory_seq_rm(mem, s, -1, -1);
            round++;
        }
        lk.unlock();
        rx.join();
        printf("RESULT mode=drafter g=%d branches=%d group=%d replica=%d:%d rounds=%d matches=%d partial=%d early=%d sent=%d catchups=%d branch_batch=%.2f\n",
               g, n_branch, n_group, rep_id, n_rep, n_rounds, n_match, n_partial, n_early, n_sent, n_catchup,
               n_branch_steps ? (double) n_branch_tok / n_branch_steps : 0.0);
        fflush(stdout);
        return 0;
    }

    if (mode == "drafter") {
        // ---- drafter on its own board
        const int gam = g - 1;
        runner dft;
        dft.load(md_path.empty() ? m_path : md_path, n_ctx, n_threads);
        std::stringstream ss; ss << std::ifstream(f_path).rdbuf();
        const std::string prompt = ss.str();
        std::vector<llama_token> confirmed(prompt.size() + 16);
        const int n_prompt = llama_tokenize(dft.vocab, prompt.c_str(), (int) prompt.size(), confirmed.data(), (int) confirmed.size(), true, true);
        confirmed.resize(n_prompt);
        dft.decode(std::vector<llama_token>(confirmed.begin(), confirmed.end() - 1));
        const int fd = listen_accept(listen_port);

        std::vector<llama_token> ahead;          // drafted beyond `confirmed`; KV holds (confirmed + ahead)[0, n_past)
        auto extend = [&]() {
            std::vector<llama_token> stream = confirmed;
            stream.insert(stream.end(), ahead.begin(), ahead.end());
            dft.decode(std::vector<llama_token>(stream.begin() + dft.n_past, stream.end()));
            ahead.push_back(dft.top(-1));
        };
        int n_rounds = 0, n_hit = 0, n_sent_ready = 0;
        double ms_block = 0; // time between a result arriving and the next block being sent
        auto t_res = clk::now();
        while (true) {
            if ((int) ahead.size() >= gam) n_sent_ready++;
            while ((int) ahead.size() < gam) extend();
            if (n_rounds > 0) ms_block += ms_since(t_res);
            d2v blk = {}; blk.round = n_rounds;
            for (int i = 0; i < gam; ++i) blk.toks[i] = ahead[i];
            send_all(fd, &blk, sizeof(blk));
            // keep drafting ahead (guess: whole block accepted, bonus = our own next token) until the result arrives
            v2d r;
            while (true) {
                pollfd pf = { fd, POLLIN, 0 };
                if ((int) ahead.size() < 2 * gam + 1 && poll(&pf, 1, 0) == 0) { extend(); continue; }
                recv_all(fd, &r, sizeof(r));
                if (r.type == 0) continue; // forward of our own block
                break;
            }
            t_res = clk::now();
            n_rounds++;
            if (r.stop) break;
            const int L_old = (int) confirmed.size();
            if (r.n_acc == gam && (int) ahead.size() > gam && ahead[gam] == r.bonus) {
                n_hit++;
                confirmed.insert(confirmed.end(), ahead.begin(), ahead.begin() + gam + 1);
                ahead.erase(ahead.begin(), ahead.begin() + gam + 1);
            } else {
                confirmed.insert(confirmed.end(), ahead.begin(), ahead.begin() + r.n_acc);
                confirmed.push_back(r.bonus);
                ahead.clear();
                dft.truncate(std::min(dft.n_past, L_old + r.n_acc));
            }
        }
        printf("RESULT mode=drafter g=%d rounds=%d hits=%d blocks_ready=%d mean_block_ms=%.0f\n",
               g, n_rounds, n_hit, n_sent_ready, n_rounds > 1 ? ms_block / (n_rounds - 1) : 0.0);
        fflush(stdout);
        return 0;
    }

    int rank = 0, world = 1;
    if (mode == "ppv") {
        if (!llama_ppv_init_from_env()) { fprintf(stderr, "ppv: set LLAMA_PPV_PEERS and LLAMA_PPV_RANK\n"); return 1; }
        rank  = llama_ppv_get_rank();
        world = llama_ppv_get_world();
        g = world;
    }
    // remote drafter: ppv rank 0, or sd (= control C: drafter board + one batched-verify board)
    const bool remote   = (mode == "ppv" || mode == "sd") && rank == 0 && !drafter_addr.empty();
    const bool drafting = mode != "ar" && rank == 0 && !remote;
    if (drafting && md_path.empty()) { fprintf(stderr, "-md required for %s\n", mode.c_str()); return 1; }

    runner tgt, dft;
    tgt.load(m_path, n_ctx, n_threads);
    if (drafting) dft.load(md_path, n_ctx, n_threads);

    std::stringstream ss; ss << std::ifstream(f_path).rdbuf();
    const std::string prompt = ss.str();
    std::vector<llama_token> seq(prompt.size() + 16);
    const int n_prompt = llama_tokenize(tgt.vocab, prompt.c_str(), (int) prompt.size(), seq.data(), (int) seq.size(), true, true);
    if (n_prompt <= 1) { fprintf(stderr, "tokenize failed\n"); return 1; }
    seq.resize(n_prompt);

    // prefill everything but the last prompt token (replicated on every rank)
    const auto t_pre = clk::now();
    tgt.decode(std::vector<llama_token>(seq.begin(), seq.end() - 1));
    if (drafting) dft.decode(std::vector<llama_token>(seq.begin(), seq.end() - 1));
    const double ms_pre = ms_since(t_pre);

    std::vector<int> dfds; // drafter replicas
    if (remote) {
        std::stringstream as(drafter_addr); std::string hp;
        while (std::getline(as, hp, ',')) dfds.push_back(connect_retry(hp));
    }
    int d_round = 0;
    if (mode == "ppv") { // align ranks before timing
        int32_t x = 0; std::vector<int32_t> all(world);
        llama_ppv_allgather(&x, sizeof(x), all.data());
    }

    const int gam = g - 1;
    int n_gen = 0, n_rounds = 0, n_acc_total = 0;
    double ms_draft = 0, ms_verify = 0;
    const auto t0 = clk::now();
    bool eog = false;
    while (n_gen < n_predict && !eog) {
        std::vector<llama_token> drafts;
        if (mode == "ar") {
            const auto tv = clk::now();
            tgt.decode({ seq.back() });
            const llama_token t = tgt.top(0);
            ms_verify += ms_since(tv);
            seq.push_back(t); n_gen++; n_rounds++;
            eog = llama_vocab_is_eog(tgt.vocab, t);
            continue;
        }

        // ---- draft gam tokens (rank 0)
        const auto td = clk::now();
        if (drafting) {
            dft.decode(std::vector<llama_token>(seq.begin() + dft.n_past, seq.end()));
            drafts.push_back(dft.top(-1));
            while ((int) drafts.size() < gam) {
                dft.decode({ drafts.back() });
                drafts.push_back(dft.top(-1));
            }
        }
        if (remote) { // first block for this round from any replica wins; stale duplicates are dropped
            d2v blk = {}; bool got = false;
            std::vector<pollfd> pf(dfds.size());
            while (!got) {
                for (size_t i = 0; i < dfds.size(); ++i) pf[i] = { dfds[i], POLLIN, 0 };
                poll(pf.data(), pf.size(), -1);
                for (size_t i = 0; i < dfds.size() && !got; ++i) {
                    if (!(pf[i].revents & POLLIN)) continue;
                    d2v m; recv_all(dfds[i], &m, sizeof(m));
                    if (m.round == d_round) { blk = m; got = true; }
                }
            }
            v2d f = {}; f.type = 0; f.round = d_round;
            for (int i = 0; i < gam; ++i) f.toks[i] = blk.toks[i];
            for (int x : dfds) send_all(x, &f, sizeof(f));
            drafts.assign(blk.toks, blk.toks + gam);
        }
        if (mode == "ppv") { // rank 0's drafts to everyone
            std::vector<int32_t> mine(gam, 0), all((size_t) gam * world);
            if (rank == 0) for (int i = 0; i < gam; ++i) mine[i] = drafts[i];
            llama_ppv_allgather(mine.data(), gam * sizeof(int32_t), all.data());
            drafts.assign(all.begin(), all.begin() + gam);
        }
        ms_draft += ms_since(td);

        // ---- verify g = gam + 1 positions
        const auto tv = clk::now();
        std::vector<llama_token> vb = { seq.back() };
        vb.insert(vb.end(), drafts.begin(), drafts.end());
        std::vector<llama_token> preds(g);
        if (mode == "sd") {
            std::vector<int> rows(g); for (int i = 0; i < g; ++i) rows[i] = i;
            tgt.decode(vb, rows);
            for (int i = 0; i < g; ++i) preds[i] = tgt.top(i);
        } else {
            llama_ppv_set_row(rank);
            tgt.decode(vb, { rank });
            llama_ppv_set_row(-1);
            int32_t mine = tgt.top(rank);
            std::vector<int32_t> all(world);
            llama_ppv_allgather(&mine, sizeof(mine), all.data());
            for (int i = 0; i < g; ++i) preds[i] = all[i];
        }
        ms_verify += ms_since(tv);

        // ---- greedy acceptance
        const int base = (int) seq.size(); // tgt cache holds [0, base-1); vb sits at base-1 .. base-1+gam
        int n_acc = 0;
        while (n_acc < gam && drafts[n_acc] == preds[n_acc]) n_acc++;
        for (int i = 0; i < n_acc && !eog; ++i) {
            seq.push_back(drafts[i]); n_gen++;
            eog = llama_vocab_is_eog(tgt.vocab, drafts[i]);
        }
        if (!eog) {
            seq.push_back(preds[n_acc]); n_gen++;
            eog = llama_vocab_is_eog(tgt.vocab, preds[n_acc]);
        }
        tgt.truncate(base + n_acc);
        if (drafting) dft.truncate(std::min(dft.n_past, base + n_acc));
        n_rounds++; n_acc_total += n_acc;
        if (remote) {
            v2d r = {}; r.type = 1; r.round = d_round++; r.n_acc = n_acc; r.bonus = preds[n_acc]; r.stop = (n_gen >= n_predict || eog) ? 1 : 0;
            for (int x : dfds) send_all(x, &r, sizeof(r));
        }
    }
    const double ms = ms_since(t0);

    uint64_t h = 1469598103934665603ULL;
    std::string text;
    // hash/text over exactly the first n_predict generated tokens, so all modes compare directly
    for (int i = n_prompt; i < (int) seq.size() && i < n_prompt + n_predict; ++i) {
        h = (h ^ (uint64_t) seq[i]) * 1099511628211ULL;
        char buf[256]; int n = llama_token_to_piece(tgt.vocab, seq[i], buf, sizeof(buf), 0, false);
        if (n > 0) text.append(buf, n);
    }
    int64_t pr = 0; double pw = 0;
    if (mode == "ppv") llama_ppv_stats(&pr, &pw);
    printf("RESULT mode=%s%s rank=%d world=%d g=%d prompt=%d gen=%d rounds=%d acc/round=%.2f tok/round=%.2f "
           "ms=%.0f tok/s=%.3f draft_ms=%.0f verify_ms=%.0f prefill_ms=%.0f ppv_rounds=%lld ppv_wait_ms=%.0f hash=%016llx\n",
           mode.c_str(), remote ? "+remote" : "", rank, world, g, n_prompt, n_gen, n_rounds, n_rounds ? (double) n_acc_total / n_rounds : 0.0,
           n_rounds ? (double) n_gen / n_rounds : 0.0, ms, n_gen / (ms / 1000.0), ms_draft, ms_verify, ms_pre,
           (long long) pr, pw, (unsigned long long) h);
    if (rank == 0) printf("TEXT %s\n", text.c_str());
    fflush(stdout);
    return 0;
}
