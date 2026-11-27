// Vocab-only native oracle. It deliberately loads only the CPU backend.
#include "ggml-backend.h"
#include "llama.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
static void check(bool ok, const char* reason) { if (!ok) throw std::runtime_error(reason); }
static std::string hex(const std::string& x) {
    static const char digits[]="0123456789abcdef";
    std::string out;
    for (unsigned char c:x) { out+=digits[c>>4]; out+=digits[c&15]; }
    return out;
}
static std::string unhex(const std::string& x) {
    check(x.size()%2==0,"bad hex fixture");
    std::string out;
    for (size_t i=0; i<x.size(); i+=2) out+=char(std::stoul(x.substr(i,2),nullptr,16));
    return out;
}
int main(int argc,char** argv) {
    if (argc!=4) return 2;
    try {
        check(ggml_backend_load("/opt/homebrew/opt/ggml/libexec/libggml-cpu-apple_m4.so"),"CPU backend");
        auto mp=llama_model_default_params(); mp.n_gpu_layers=0;mp.vocab_only=true;
        auto model=llama_model_load_from_file(argv[1],mp);check(model,"model vocab load");
        auto v=llama_model_get_vocab(model);auto n=llama_vocab_n_tokens(v);
        check(n==128256,"vocabulary width");fs::path out=argv[3];
        std::ifstream inputs(argv[2]);check(inputs.good(),"input manifest");
        std::ofstream eval(out/"oracle.tsv");eval.exceptions(std::ios::badbit|std::ios::failbit);
        eval<<"case\tadd_bos\tparse_special\tinput_utf8\tids\tdecoded_hidden\tdecoded_visible\n";
        std::string line;
        while (std::getline(inputs,line)) {
            auto tab=line.find('\t');check(tab!=line.npos,"case format");
            auto text=unhex(line.substr(tab+1));
            for(int bos:{0,1})for(int parse:{0,1}){
                std::vector<int> ids(text.size()+3);
                int count=llama_tokenize(v,text.data(),int(text.size()),ids.data(),int(ids.size()),bos,parse);
                check(count>=0,"native tokenize");ids.resize(count);
                eval<<line.substr(0,tab)<<'\t'<<bos<<'\t'<<parse<<'\t'<<hex(text)<<'\t';
                for(size_t i=0;i<ids.size();i++){if(i)eval<<',';eval<<ids[i];}
                for(int display:{0,1}){
                    std::vector<char> bytes(ids.size()*160+32);
                    int bytes_used=llama_detokenize(v,ids.data(),int(ids.size()),bytes.data(),int(bytes.size()),false,display);
                    check(bytes_used>=0,"native detokenize");eval<<'\t'<<hex(std::string(bytes.data(),bytes_used));
                }
                eval<<'\n';
            }
        }
        for(int display:{0,1}){
            std::ofstream pieces(out/("pieces"+std::to_string(display)+".bin"),std::ios::binary);
            pieces.exceptions(std::ios::badbit|std::ios::failbit);
            for(int i=0;i<n;i++){
                std::vector<char> bytes(1024);
                int used=llama_token_to_piece(v,i,bytes.data(),int(bytes.size()),0,display);
                check(used>=0,"native token piece");
                for(int shift:{0,8,16,24}) pieces.put(char((used>>shift)&255));
                pieces.write(bytes.data(),used);
            }
        }
        std::ofstream eog(out/"eog.txt");
        for(int i=0;i<n;i++) if(llama_vocab_is_eog(v,i))eog<<i<<'\n';
        llama_model_free(model);
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
