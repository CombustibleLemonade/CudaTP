#include <vector>
#include <string>
#include <chrono>

uint64_t time_since_epoch_milliseconds() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

std::vector<std::string> problems{
    "l103_sincos10", 
    "t56_topgen_1",
    "t155_xreal_1", 
    "t5_jordan5a", 
    "t30_borsuk_4", 
    "l26_arytm_1", 
    "t5_square_1", 
    "t28_complex1", 
    "t161_xxreal_1", 
    "t50_funct_8", 
    "t17_mesfunc5", 
    "t49_member_1", 
    "t56_topgen_1", 
    "t149_zf_lang1", 
    "t18_rlvect_2", 
    "t151_finseq_3", 
    "t48_xreal_1", 
    "t3_seq_4", 
    "t174_relat_1", 
    "t18_simplex0", 
    "t8_mcart_1", 
    "t119_gfacirc1", 
    "t1_subset_1", 
    "l9_topalg_5", 
    "l32_ndiff_3", 
    "t114_flang_2"};