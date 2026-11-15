#include "llama-ppv.h"

#include "ggml.h"
#include "llama-impl.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

struct ppv_state {
    bool enabled = false;
    int  rank    = 0;
    int  world   = 1;
    int  row     = -1;
    std::vector<std::string> peers;
    std::vector<int>         fds;
    std::vector<uint8_t>     buf;
    int64_t n_rounds = 0;
    double  wait_ms  = 0.0;
};

ppv_state & st() {
    static ppv_state s;
    return s;
}

void split_host_port(const std::string & hp, std::string & host, int & port) {
    size_t c = hp.rfind(':');
    GGML_ASSERT(c != std::string::npos);
    host = hp.substr(0, c);
    port = atoi(hp.c_str() + c + 1);
}

void send_all(int fd, const void * buf, size_t n) {
    const char * p = (const char *) buf;
    while (n > 0) {
        ssize_t w = send(fd, p, n, 0);
        GGML_ASSERT(w > 0 && "ppv: send failed");
        p += w; n -= (size_t) w;
    }
}

void recv_all(int fd, void * buf, size_t n) {
    char * p = (char *) buf;
    while (n > 0) {
        ssize_t r = recv(fd, p, n, 0);
        GGML_ASSERT(r > 0 && "ppv: recv failed");
        p += r; n -= (size_t) r;
    }
}

void set_nodelay(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

// full mesh: connect to lower ranks, accept from higher ranks
void connect_mesh() {
    auto & s = st();
    s.fds.assign(s.world, -1);

    std::string host; int port;
    split_host_port(s.peers[s.rank], host, port);
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr = {};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port        = htons(port);
    GGML_ASSERT(bind(lfd, (sockaddr *) &addr, sizeof(addr)) == 0 && "ppv: bind failed");
    GGML_ASSERT(listen(lfd, s.world) == 0);

    for (int p = 0; p < s.rank; ++p) {
        std::string h; int pt;
        split_host_port(s.peers[p], h, pt);
        addrinfo hints = {}, * res = nullptr;
        hints.ai_family   = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        GGML_ASSERT(getaddrinfo(h.c_str(), std::to_string(pt).c_str(), &hints, &res) == 0);
        int fd = -1;
        for (int attempt = 0; attempt < 6000; ++attempt) { // up to 10 min (peers may still be loading)
            fd = socket(AF_INET, SOCK_STREAM, 0);
            if (connect(fd, res->ai_addr, res->ai_addrlen) == 0) {
                break;
            }
            close(fd); fd = -1;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        freeaddrinfo(res);
        GGML_ASSERT(fd >= 0 && "ppv: connect failed");
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

void gather_kv_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    GGML_UNUSED(userdata);
    if (ith != 0) {
        return;
    }
    const ggml_tensor * k = dst->src[0];
    const ggml_tensor * v = dst->src[1];
    GGML_ASSERT(k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(k) && ggml_is_contiguous(v));
    const size_t nk = ggml_nbytes(k);
    const size_t nv = ggml_nbytes(v);
    GGML_ASSERT(dst->nb[1] == nk + nv);

    std::vector<uint8_t> mine(nk + nv);
    memcpy(mine.data(),      k->data, nk);
    memcpy(mine.data() + nk, v->data, nv);
    llama_ppv_allgather(mine.data(), nk + nv, dst->data);
}

} // namespace

bool llama_ppv_init_from_env() {
    auto & s = st();
    if (s.enabled) {
        return true;
    }
    const char * peers = getenv("LLAMA_PPV_PEERS");
    if (!peers || !*peers) {
        return false;
    }
    std::string p = peers;
    size_t pos = 0;
    while (pos <= p.size()) {
        size_t e = p.find(',', pos);
        if (e == std::string::npos) {
            e = p.size();
        }
        if (e > pos) {
            s.peers.push_back(p.substr(pos, e - pos));
        }
        pos = e + 1;
    }
    s.world = (int) s.peers.size();
    const char * r = getenv("LLAMA_PPV_RANK");
    s.rank = r ? atoi(r) : 0;
    GGML_ASSERT(s.rank >= 0 && s.rank < s.world);
    if (s.world > 1) {
        connect_mesh();
    }
    s.enabled = true;
    return true;
}

int32_t llama_ppv_get_rank()  { return st().rank; }
int32_t llama_ppv_get_world() { return st().world; }

void llama_ppv_set_row(int32_t row) { st().row = row; }
int32_t llama_ppv_get_row()         { return st().row; }

void llama_ppv_allgather(const void * src, size_t n, void * dst) {
    auto & s = st();
    const auto t0 = std::chrono::steady_clock::now();
    memcpy((uint8_t *) dst + (size_t) s.rank * n, src, n);
    for (int p = 0; p < s.world; ++p) {
        if (p != s.rank) {
            send_all(s.fds[p], src, n);
        }
    }
    for (int p = 0; p < s.world; ++p) {
        if (p != s.rank) {
            recv_all(s.fds[p], (uint8_t *) dst + (size_t) p * n, n);
        }
    }
    s.wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    s.n_rounds++;
}

ggml_tensor * llama_ppv_gather_kv(ggml_context * ctx, ggml_tensor * k, ggml_tensor * v) {
    ggml_tensor * args[2] = { k, v };
    const int64_t n = ggml_nelements(k) + ggml_nelements(v);
    return ggml_custom_4d(ctx, GGML_TYPE_F32, n, st().world, 1, 1, args, 2, gather_kv_op, 1, nullptr);
}

void llama_ppv_stats(int64_t * n_rounds, double * wait_ms) {
    *n_rounds = st().n_rounds;
    *wait_ms  = st().wait_ms;
}
