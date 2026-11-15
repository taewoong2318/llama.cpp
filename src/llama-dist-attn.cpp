#include "llama-dist-attn.h"

#include "llama-impl.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>


#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

struct dist_state {
    bool init = false;
    bool enabled = false;
    int  rank  = 0;
    int  world = 1;
    std::vector<std::string> peers;
    std::vector<int> fds;
    int64_t own_lo = 0;
    int64_t own_hi = std::numeric_limits<int64_t>::max();
    float   kq_scale = 1.0f;

    bool    stats = false;
    int64_t n_merge = 0;
    std::vector<float> t_net_us; // per merge: send + wait for all peers
};

dist_state & st() {
    static dist_state s;
    return s;
}

void parse_env() {
    auto & s = st();
    if (s.init) {
        return;
    }
    s.init = true;
    const char * peers = getenv("LLAMA_DIST_PEERS");
    if (!peers || !*peers) {
        return;
    }
    std::string p = peers;
    size_t pos = 0;
    while (pos <= p.size()) {
        size_t e = p.find(',', pos);
        if (e == std::string::npos) e = p.size();
        if (e > pos) s.peers.push_back(p.substr(pos, e - pos));
        pos = e + 1;
    }
    s.world = (int) s.peers.size();
    const char * r = getenv("LLAMA_DIST_RANK");
    s.rank = r ? atoi(r) : 0;
    GGML_ASSERT(s.rank >= 0 && s.rank < s.world);
    const char * own = getenv("LLAMA_DIST_OWN");
    if (own && *own) {
        const char * c = strchr(own, ':');
        GGML_ASSERT(c);
        s.own_lo = atoll(own);
        if (*(c + 1)) s.own_hi = atoll(c + 1);
    }
    s.stats = getenv("LLAMA_DIST_STATS") != nullptr;
    s.enabled = true;
    if (s.stats) {
        atexit([] {
            auto & s = st();
            // skip the first 30 merges (first token), which absorb start-up skew between ranks
            if (s.t_net_us.size() > 60) {
                std::vector<float> t(s.t_net_us.begin() + 30, s.t_net_us.end());
                std::sort(t.begin(), t.end());
                fprintf(stderr, "dist_attn: rank %d/%d, %lld merges, net us: median %.0f p90 %.0f max %.0f\n",
                        s.rank, s.world, (long long) s.n_merge, t[t.size() / 2], t[t.size() * 9 / 10], t.back());
            }
        });
    }
}

void split_host_port(const std::string & hp, std::string & host, int & port) {
    size_t c = hp.rfind(':');
    GGML_ASSERT(c != std::string::npos);
    host = hp.substr(0, c);
    port = atoi(hp.c_str() + c + 1);
}

void set_nodelay(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

void send_all(int fd, const void * buf, size_t n) {
    const char * p = (const char *) buf;
    while (n > 0) {
        ssize_t w = send(fd, p, n, 0);
        GGML_ASSERT(w > 0 && "dist_attn: send failed");
        p += w; n -= (size_t) w;
    }
}

void recv_all(int fd, void * buf, size_t n) {
    char * p = (char *) buf;
    while (n > 0) {
        ssize_t r = recv(fd, p, n, 0);
        GGML_ASSERT(r > 0 && "dist_attn: recv failed");
        p += r; n -= (size_t) r;
    }
}

// full mesh: connect to lower ranks, accept from higher ranks
void connect_mesh() {
    auto & s = st();
    if (s.world == 1 || !s.fds.empty()) {
        return;
    }
    s.fds.assign(s.world, -1);

    std::string host; int port;
    split_host_port(s.peers[s.rank], host, port);
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    GGML_ASSERT(bind(lfd, (sockaddr *) &addr, sizeof(addr)) == 0 && "dist_attn: bind failed");
    GGML_ASSERT(listen(lfd, s.world) == 0);

    for (int p = 0; p < s.rank; ++p) {
        std::string h; int pt;
        split_host_port(s.peers[p], h, pt);
        addrinfo hints = {}, * res = nullptr;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        GGML_ASSERT(getaddrinfo(h.c_str(), std::to_string(pt).c_str(), &hints, &res) == 0);
        int fd = -1;
        // peers may still be prefilling (minutes apart on long prompts): retry for up to 1 h
        for (int attempt = 0; attempt < 36000; ++attempt) {
            fd = socket(AF_INET, SOCK_STREAM, 0);
            if (connect(fd, res->ai_addr, res->ai_addrlen) == 0) break;
            close(fd); fd = -1;
            if (attempt % 600 == 599) {
                LLAMA_LOG_INFO("%s: rank %d still waiting for rank %d\n", __func__, s.rank, p);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        freeaddrinfo(res);
        GGML_ASSERT(fd >= 0 && "dist_attn: connect failed");
        set_nodelay(fd);
        int32_t me = s.rank;
        send_all(fd, &me, sizeof(me));
        s.fds[p] = fd;
    }
    for (int n = s.rank + 1; n < s.world; ++n) {
        int fd = accept(lfd, nullptr, nullptr);
        GGML_ASSERT(fd >= 0);
        set_nodelay(fd);
        int32_t who = -1;
        recv_all(fd, &who, sizeof(who));
        GGML_ASSERT(who > s.rank && who < s.world && s.fds[who] < 0);
        s.fds[who] = fd;
    }
    close(lfd);
    LLAMA_LOG_INFO("%s: rank %d/%d connected\n", __func__, s.rank, s.world);
}

inline float mask_at(const ggml_tensor * mask, int64_t j) {
    if (mask->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(((const ggml_fp16_t *) mask->data)[j]);
    }
    return ((const float *) mask->data)[j];
}

// src0: kq [n_own, 1, n_head] f32, src1: mask view [n_own, 1] f16/f32
// dst [n_own + 2, 1, n_head]: p = exp(scale*kq + mask - m), then m and s = sum(p) per head
void exp_partial(ggml_tensor * dst, int ith, int nth, void * /*userdata*/) {
    const auto & s = st();
    const ggml_tensor * kq   = dst->src[0];
    const ggml_tensor * mask = dst->src[1];
    const int64_t n = kq->ne[0];
    const int n_head = (int) kq->ne[2];
    for (int h = ith; h < n_head; h += nth) {
        const float * x = (const float *) ((const char *) kq->data + h * kq->nb[2]);
        float * p = (float *) ((char *) dst->data + h * dst->nb[2]);
        float m = -INFINITY;
        for (int64_t j = 0; j < n; ++j) {
            p[j] = x[j] * s.kq_scale + mask_at(mask, j);
            m = std::max(m, p[j]);
        }
        float sum = 0.0f;
        for (int64_t j = 0; j < n; ++j) {
            p[j] = std::isfinite(m) ? expf(p[j] - m) : 0.0f; // exp(-inf) = 0 for masked cells
            sum += p[j];
        }
        p[n] = m; p[n + 1] = sum;
    }
}

// merge rows (o[hd], m, s) in order
void lse_merge(const float * rows, int n, int64_t stride, int hd, float * o, float & m, float & s) {
    m = -INFINITY;
    for (int i = 0; i < n; ++i) {
        const float * r = rows + i * stride;
        if (r[hd + 1] > 0.0f) m = std::max(m, r[hd]);
    }
    s = 0.0f;
    memset(o, 0, hd * sizeof(float));
    if (!std::isfinite(m)) return;
    for (int i = 0; i < n; ++i) {
        const float * r = rows + i * stride;
        if (r[hd + 1] <= 0.0f) continue;
        const float c = expf(r[hd] - m);
        s += r[hd + 1] * c;
        for (int d = 0; d < hd; ++d) o[d] += r[d] * c;
    }
}

// src0: kqv [hd, 1, n_head] (unnormalized), src1: exp_partial dst (m, s at [n_own], [n_own+1])
// dst [hd*n_head, 1]
void attn_merge(ggml_tensor * dst, int ith, int /*nth*/, void * /*userdata*/) {
    if (ith != 0) return;
    auto & s = st();
    const ggml_tensor * kqv = dst->src[0];
    const ggml_tensor * ms  = dst->src[1];
    const int hd = (int) kqv->ne[0], n_head = (int) kqv->ne[2], W = hd + 2;
    const int64_t n_own = ms->ne[0] - 2;

    std::vector<float> mine(n_head * W);
    for (int h = 0; h < n_head; ++h) {
        float * r = mine.data() + h * W;
        const char * o = (const char *) kqv->data + h * kqv->nb[2];
        for (int d = 0; d < hd; ++d) r[d] = *(const float *) (o + d * kqv->nb[0]);
        const float * mr = (const float *) ((const char *) ms->data + h * ms->nb[2]);
        r[hd] = mr[n_own]; r[hd + 1] = mr[n_own + 1];
    }

    std::vector<float> all(s.world * n_head * W);
    memcpy(all.data() + s.rank * n_head * W, mine.data(), mine.size() * sizeof(float));
    if (s.world > 1) {
        connect_mesh();
        const auto t0 = std::chrono::steady_clock::now();
        const size_t nbytes = mine.size() * sizeof(float);
        for (int p = 0; p < s.world; ++p) {
            if (p != s.rank) send_all(s.fds[p], mine.data(), nbytes);
        }
        for (int p = 0; p < s.world; ++p) {
            if (p != s.rank) recv_all(s.fds[p], all.data() + p * n_head * W, nbytes);
        }
        if (s.stats) {
            s.t_net_us.push_back(std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - t0).count());
        }
    }
    s.n_merge++;

    // merge ranks in rank order, so every rank computes bit-identical results
    float * out = (float *) dst->data;
    std::vector<float> rr(s.world * W);
    for (int h = 0; h < n_head; ++h) {
        for (int r = 0; r < s.world; ++r) {
            memcpy(rr.data() + r * W, all.data() + (r * n_head + h) * W, W * sizeof(float));
        }
        float m, sum;
        float * oh = out + h * hd;
        lse_merge(rr.data(), s.world, W, hd, oh, m, sum);
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int d = 0; d < hd; ++d) oh[d] *= inv;
    }
}

} // namespace

bool llama_dist_attn_enabled() {
    parse_env();
    return st().enabled;
}

ggml_tensor * llama_dist_attn(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                              ggml_tensor * kq_mask, float kq_scale) {
    GGML_ASSERT(q->type == GGML_TYPE_F32 && q->ne[2] == 1);
    auto & s = st();
    s.kq_scale = kq_scale;

    // this rank's contiguous cell range within the current KV view
    const int64_t n_kv = k->ne[2];
    const int64_t r0 = std::min<int64_t>(std::max<int64_t>(s.own_lo, 0), n_kv);
    const int64_t r1 = std::min<int64_t>(std::max<int64_t>(s.own_hi, 0), n_kv);
    GGML_ASSERT(r1 > r0 && "dist_attn: rank owns no cells in this KV view");
    const int64_t n_own = r1 - r0;
    const bool v_trans = v->nb[1] > v->nb[2];

    // k: [hd, n_head_kv, n_kv]; v: [hd, n_head_kv, n_kv] or transposed [n_kv, n_head_kv, hd]
    ggml_tensor * ko = ggml_view_3d(ctx, k, k->ne[0], k->ne[1], n_own, k->nb[1], k->nb[2], r0 * k->nb[2]);
    ggml_tensor * vo = v_trans
        ? ggml_view_3d(ctx, v, n_own, v->ne[1], v->ne[2], v->nb[1], v->nb[2], r0 * v->nb[0])
        : ggml_view_3d(ctx, v, v->ne[0], v->ne[1], n_own, v->nb[1], v->nb[2], r0 * v->nb[2]);
    ggml_tensor * mo = ggml_view_2d(ctx, kq_mask, n_own, 1, kq_mask->nb[1], r0 * ggml_element_size(kq_mask));

    // same matmuls as the non-flash path in build_attn_mha, over the owned cells only
    ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);   // [hd, 1, n_head]
    ko = ggml_permute(ctx, ko, 0, 2, 1, 3);                // [hd, n_own, n_head_kv]
    vo = ggml_permute(ctx, vo, 0, 2, 1, 3);
    ggml_tensor * kq = ggml_mul_mat(ctx, ko, qp);          // [n_own, 1, n_head]
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

    ggml_tensor * eargs[2] = { kq, mo };
    ggml_tensor * pe = ggml_custom_4d(ctx, GGML_TYPE_F32, n_own + 2, 1, kq->ne[2], 1, eargs, 2, exp_partial, GGML_N_TASKS_MAX, nullptr);
    ggml_tensor * p  = ggml_view_3d(ctx, pe, n_own, 1, pe->ne[2], pe->nb[1], pe->nb[2], 0);

    if (!v_trans) {
        vo = ggml_cont(ctx, ggml_transpose(ctx, vo));
    }
    ggml_tensor * kqv = ggml_mul_mat(ctx, vo, p);          // [hd, 1, n_head], unnormalized

    ggml_tensor * margs[2] = { kqv, pe };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, q->ne[0] * q->ne[1], 1, 1, 1, margs, 2, attn_merge, 1, nullptr);
}
