# llama.cpp with some changes

* various performance improvements for qwen4exp
  * `-sm tensor` enabled
  * mtp enabled
* ling3 is now supported in `-sm tensor`

## Split mode tensor

* --max-tensor-split to reduce tensor split to fewer gpus
  * eg. `-sm tensor -max-tensor-split 2` will only split tensors across 2 gpus, even if 10 are available
* faster load time by using async uploads during tensor split planning