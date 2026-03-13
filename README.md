> [!IMPORTANT]
> FlexB is still work-in-progress, please don't expect stability. Newer versions *will* break compatibility.

# Installation 

You just download `flexb.lua` and place it in the directory your script will access through.

For Roblox, right-click ReplicatedFirst in the Explorer tab, select Insert From File and select `flexb.lua`.

# Example use 

```lua
-- XOR example

local nn = require("flexb")
local mabs,msqrt,mlog = math.abs, math.sqrt, math.log

local ai = nn.new({2,2,1},{nn.swish,nn.linear})

local ds = {
	[{0,0}] = {0},
	[{0,1}] = {1},
	[{1,0}] = {1},
	[{1,1}] = {0}
}

local clk = os.clock()
for i=1,1000 do
	for k,v in next,ds do
		local outp,trin = ai:forward(k)
		ai:backward(trin,nn.huberderiv(outp,v),0.01)
		print(nn.huber(outp,v))
	end
end
print("\nTime taken:",os.clock()-clk)

local c = 0
for k,v in next,ds do
	local outp = ai:forward(k)
	c = c + nn.huber(outp,v)
end

print("\nAI's total inaccuracy:",c) -- likely <1e-5

print("\n#1.1.",ai.weight[1][1][1],ai.weight[1][1][2],ai.bias[1][1],ai.gamma[1][1],ai.beta[1][1])
print("#1.2.",ai.weight[1][2][1],ai.weight[1][2][2],ai.bias[1][2],ai.gamma[1][2],ai.beta[1][2])
print("#2.1.",ai.weight[2][1][1],ai.weight[2][1][2],ai.bias[2][1],ai.gamma[2][1],ai.beta[2][1])
```
