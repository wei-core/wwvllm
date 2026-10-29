"""Compare the existing kernel's tile choices using CUDA event timing."""
import json, argparse, torch, triton
from wwvllm.quantization.linear import QuantizedLinear, _quant_gemm
p=argparse.ArgumentParser(); p.add_argument('--output',required=True); args=p.parse_args()
torch.manual_seed(20261029)
rows=[]
for bits in (4,8):
 for m,n,k in ((64,1024,1024),(64,1024,3072),(512,1024,1024),(512,1024,3072)):
  q=QuantizedLinear.from_float(torch.nn.Linear(k,n,bias=False,device='cuda',dtype=torch.bfloat16),bits,128)
  x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16); y=torch.empty((m,n),device='cuda',dtype=x.dtype)
  reference=torch.nn.functional.linear(x,q.dequantize(x.dtype))
  for bm,bn,bk in ((16,32,32),(32,32,32)):
   def f():
    _quant_gemm[(triton.cdiv(m,bm),triton.cdiv(n,bn))](x,q.qweight,q.scales,q.zeros,q.g_idx,y,m,n,k,128,q.scales.shape[1],bits==4,bm,bn,bk)
   f(); torch.cuda.synchronize()
   error=(y.float()-reference.float()).abs().max().item()
   ms=triton.testing.do_bench(f,warmup=100,rep=300)
   row=dict(bits=bits,m=m,n=n,k=k,bm=bm,bn=bn,bk=bk,ms=ms,max_abs_error=error)
   rows.append(row); print(json.dumps(row),flush=True)
with open(args.output,'w') as f: json.dump(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,triton=triton.__version__,results=rows),f,indent=2)
