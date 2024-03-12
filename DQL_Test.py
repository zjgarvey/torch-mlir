import numpy as np
from iree import compiler as ic
from iree import runtime as rt

with open('DQL.torch.mlir', 'r') as f:
    IR_String = f.read()

compiled_fb = ic.tools.compile_str(
    IR_String,
    target_backends=["llvm-cpu"],
)

config = rt.Config("local-task")
ctx = rt.SystemContext(config=config)
vm_module = rt.VmModule.copy_buffer(ctx.instance, compiled_fb)
ctx.add_vm_module(vm_module)

print("Invoke DQL")
arg0 = np.random.randn(6).astype(np.float32)
#arg0 = np.array([0.23692754, 1.5836138, -0.00216732, -1.5991051,   1.4991285,  -0.40704334]).astype(np.float32)
#arg0[0] = arg0[0]+25
func = ctx.modules.dql_test["test_dynamicquantizelinear"]
print("Input= ", arg0)
r =[]
for n in range(3):
    r.append(func(arg0)[n].to_host())
print("Result= (", r[0], ", ", r[1], ", ", r[2], ")")
# manually compute actual result:
max = arg0.max()
min = arg0.min()
scale = (max - min)/255
print("manually computed scale: ", scale)
def saturate(x):
    if (x < 0):
        x = 0
    if (x > 255):
        x = 255
    return x
zp = 0 - min/scale
zp = saturate(zp)
zp = round(zp)
print("manually computed zero point ", zp)
qout = arg0/scale
qout = np.round(qout)
qout = qout + zp
for i in range(6):
    qout[i] = saturate(qout[i])
print("manually computed quantized tensor ", qout)

