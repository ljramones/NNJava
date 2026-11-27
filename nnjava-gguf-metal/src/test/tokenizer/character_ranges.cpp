// Regenerate the three category transitions for the pinned llama.cpp Unicode classifier.
#include <cstdint>
#include <cstdio>
extern uint16_t unicode_cpt_flags_from_cpt(uint32_t);
int main() {
    unsigned last = 4;
    for (uint32_t c = 0; c <= 0x10ffff; ++c) {
        uint16_t flags = unicode_cpt_flags_from_cpt(c);
        unsigned kind = flags & 0x4 ? 1 : flags & 0x2 ? 2 : flags & 0x100 ? 3 : 0;
        if (kind != last) {
            std::printf("%u\n", c * 4 + kind);
            last = kind;
        }
    }
}
