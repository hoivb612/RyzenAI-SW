Forked from https://github.com/amd/RyzenAI-SW
Into https://github.com/hoivb612/RyzenAI-SW

=================================================================
REM GGML on Strix AIE

Install Anaconda 
- Update registry Session Manager/Environment system path to include "c:\ProgramData\anaconda3\Scripts"

Install RyzenAI 1.2 MSI
RYZEN_AI_INSTALLATION_PATH="C:\Program Files\RyzenAI\1.2.0\"

set XLNX_VART_FIRMWARE=%RYZEN_AI_INSTALLATION_PATH%/voe-4.0-win_amd64/xclbins/strix/AMD_AIE2P_4x4_Overlay.xclbin
set XLNX_TARGET_NAME=AMD_AIE2P_4x4_Overlay (Nx4 by default, 4x4 is performance mode)

(ryzen-ai-1.2.0) C:\llama.cpp\RyzenAI>git clone  https://github.com/hoivb612/RyzenAI-SW .

Activate ryzenai-transformers conda-enviornment
cd <transformers> (cd c:\llama.cpp\RyzenAI\example\transformers)

set TRANSFORMERS_ROOT=%CD%
conda env create --file=env.yaml
conda activate ryzenai-transformers

Use subst when path is too long
@REM use any unused drive letter, Z: for example
subst Z: %cd%

Build and Install RyzenAI
setup_stx.bat

cd %TRANSFORMERS_ROOT%\ops\cpp
cmake -B build\ -DCMAKE_INSTALL_PREFIX=%CONDA_PREFIX%
cmake --build build\ --config=Release
cmake --install build\ --config=Release

Build llama.cpp
cd %TRANSFORMERS_ROOT%\ext\llama.cpp
cmake -B build\ -DCMAKE_PREFIX_PATH="%CONDA_PREFIX%;%XRT_PATH%" -DLLAMA_RYZENAI=ON
cmake --build build\ --config=Release
Note: To switch between CPU/NPU recompile with compilation flag LLAMA_RYZENAI=OFF/ON

Build_dc_ryzen.clang: <Failed to build!!!! AVX512 instructions w/ CLang...>
cmake .. -T CLangCL -DBUILD_SHARED_LIBS=ON -DCMAKE_PREFIX_PATH="%CONDA_PREFIX%;%XRT_PATH%" -DLLAMA_RYZENAI=ON
Build_dc_ryzen.clang:   <Failed to build!!!! AVX512 instructions...>
cmake .. -T CLangCL -DBUILD_SHARED_LIBS=ON -DCMAKE_PREFIX_PATH="%CONDA_PREFIX%;%XRT_PATH%" -DLLAMA_RYZENAI=ON -DLLAMA_IQK=ON
C:\llama.cpp\llama.dc_iqk\build_dc_Ryzen.msvc: 
cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_PREFIX_PATH="%CONDA_PREFIX%;%XRT_PATH%" -DLLAMA_RYZENAI=ON
C:\llama.cpp\llama.dc_iqk\build_dc_iqk_Ryzen.msvc:
cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_PREFIX_PATH="%CONDA_PREFIX%;%XRT_PATH%" -DLLAMA_RYZENAI=ON -DLLAMA_IQK=ON
C:\llama.cpp\llama.dc_iqk\build_dc_iqk.msvc:
cmake .. -DBUILD_SHARED_LIBS=ON -DLLAMA_IQK=ON
C:\llama.cpp\llama.dc_iqk\build_dc_iqk.clang:
cmake .. -T CLangCL -DBUILD_SHARED_LIBS=ON -DLLAMA_IQK=ON
C:\llama.cpp\llama.dc_iqk\build_dc.msvc:
cmake .. -DBUILD_SHARED_LIBS=ON -DLLAMA_IQK=OFF
C:\llama.cpp\llama.dc_iqk\build_dc.clang:
cmake .. -T CLangCL -DBUILD_SHARED_LIBS=ON -DLLAMA_IQK=OFF

Note: model must be Q4_0 quantized for offload to NPU 
Example model: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_0.gguf

Run
cd %TRANSFORMERS_ROOT%\ext\llama.cpp\build\
bin\Release\main.exe -m ..\models\llama-2-7b-chat.Q4_0.gguf -e -t 1 -n 400 -p "Building a website can be done in 10 simple steps:\nStep 1:"
Example of running Llama3
Command:
bin\Release\main.exe -m ..\models\Meta-Llama-3-8b-Instruct.Q4_0.gguf -e -t 4 -n 400 -p "Building a website can be done in 10 simple steps:\nStep 1:"
Prompt:
Building a website can be done in 10 simple steps:\n
Output:
Step 1: Choose a Domain Name
Step 2: Select a Web Hosting Service
…
Step 10: Maintain and Update

Details about the quantization scheme "Q4_0" implementation on RyzenAI NPU is described in quantization.
For technical discussion of the RyzenAI backend in llama.cpp see background.

Perplexity measurement (Model quality)
The perplexity example within Llama.cpp is used to measure perplexity over a given prompt (lower perplexity is better).
The perplexity measurements in table above are done against the wikitext2 test dataset (https://paperswithcode.com/dataset/wikitext-2), with context length of 512.

cd %TRANSFORMERS_ROOT%\ext\llama.cpp\build\
bin\Release\main.exe -m ..\models\llama-2-7b-chat.Q4_0.gguf -f wikitext-2-raw\wiki.test.raw
Output:
perplexity : calculating perplexity over 655 chunks
24.43 seconds per pass - ETA 4.45 hours
[1]4.3306,[2]4.8324,[3]5.4543,[4]6.0606 ...
For more details of perplexity measurement on Llama.cpp refer to perplexity

Weights Preprocessing
The first time a matrix multiplication operator is called the weights, scales, and zero points must be statically loaded to pinned memory buffers. Additionally, the weights and scales are transposed (See Mathematics).
Mathematics
Traditional matrix multiplication (multiply row by column):
A×B=C
In llama.cpp the actual tensor layout is like this:
A×BT=CT
B is expected to already be transposed, and traspose of C is computed. This is for cache efficiency.
Additionally, the weights are typically supplied as tensor A.
RyzenAI typically expects the weights to be the B tensor in traditional matrix multiplication.
Lets use the following equation to represent what RyzenAI's matrix multiplication operator will compute:
X×W=Z
We can make use of the transpose property of matrix multiplication, which says:
BT×AT=CT
Thus if the weights are supplied by llama.cpp as A, we can forward them to RyzenAI as A^T:
X=BT
W=AT
Z=CT
Transposition of the weights is a one time penalty.

Q4_0
A weights tensor is divided into groups of 32 elements. This requires that the number of elements in a row is some multiple of 32. Every group is assigned a scaling factor that maps the floating point value to an int4 quantized value along with a zero point of 8.
To dequantize: 
y = s(q-z) 
y = float32 
weight s = float16 
scalar q = int4 
weight z = int4 zero point, and it’s always 8

==========================================================
REM To run llama-bench/kv on a regular CMD prompt: 

set DEVICE=stx
REM set PYTORCH_AIE_PATH=C:\llama.cpp\Ryzen\example\transformers\
set PYTORCH_AIE_PATH=C:\llama.test\RyzenAI

Directory of C:\llama.cpp\Ryzen\example\transformers\xclbin\stx => %PYTORCH_AIE_PATH%\xclbin\%DEVICE%\*

10/04/2024  01:02 PM    <DIR>          .
10/04/2024  01:02 PM    <DIR>          ..
10/04/2024  01:02 PM            35,051 dummy.xclbin
10/04/2024  01:02 PM         2,094,049 gemm_4x4_a16fw4acc32f.xclbin
10/04/2024  01:02 PM         1,632,580 gemm_4x4_a16w8acc64.xclbin
10/04/2024  01:02 PM         1,713,767 gemm_4x4_a8w8acc32.xclbin
10/04/2024  01:02 PM           449,670 mladf_4x4_gemm_silu_mul_a16fw4.xclbin
10/04/2024  01:02 PM         1,186,310 mladf_gemm_2x4x4_a16fw4acc16f.xclbin
10/04/2024  01:02 PM           490,394 mladf_gemm_4x4_a16fw4acc16f.xclbin

C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\* => %PYTORCH_AIE_PATH%\dll\%DEVICE%\qlinear_2\*
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_1_11008_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_1_11008_4096_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_1_4096_12288_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_1_4096_12288_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_1_4096_32768_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_1_4096_32768_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_1_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_1_4096_4096_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_32_11008_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_32_11008_4096_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_32_4096_12288_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_32_4096_12288_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_32_4096_32768_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_32_4096_32768_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_32_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_32_4096_4096_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_8_11008_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_8_11008_4096_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_8_4096_12288_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_8_4096_12288_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_8_4096_32768_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_8_4096_32768_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_8_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_8_4096_4096_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_1_11264_4096.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_1_4096_11008.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_1_4096_4096.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_32_11264_4096.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_32_4096_11008.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_32_4096_4096.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_64_11264_4096.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_64_4096_11008.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_64_4096_4096.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_8_11264_4096.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_8_4096_11008.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_8_4096_4096.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_16_2048_2048.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_16_2048_8192.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_16_8192_2048.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_1_2048_2048.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_1_2048_8192.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_1_8192_2048.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_32_2048_2048.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_32_2048_8192.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_32_8192_2048.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_64_2048_2048.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_64_2048_8192.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_64_8192_2048.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_8_2048_2048.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_8_2048_8192.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_8_8192_2048.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_128_11008_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_128_4096_12288_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_128_4096_22528_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_128_4096_32768_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_128_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_1_11008_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_1_4096_12288_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_1_4096_22528_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_1_4096_32768_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_1_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_2048_11008_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_2048_4096_12288_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_2048_4096_22528_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_2048_4096_32768_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_2048_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_128_11008_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_128_256_2048_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_128_4096_12288_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_128_4096_22528_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_128_4096_32768_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_128_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_1_11008_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_1_4096_12288_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_1_4096_22528_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_1_4096_32768_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_1_4096_32768_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_1_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_2000_11008_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_2000_4096_12288_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_2000_4096_22528_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_2000_4096_32768_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_2000_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_2048_11008_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_2048_4096_12288_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_2048_4096_22528_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_2048_4096_32768_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_2048_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_8_256_2048_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_8_256_2048_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_8_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\README.md



