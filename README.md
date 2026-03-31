> [!IMPORTANT]
> FlexB is still work-in-progress, please don't expect stability. Newer versions *will* break compatibility.

# Installation 

You just download `flexb.lua` and place it in the directory your script will access through.

For Roblox, right-click ReplicatedFirst in the Explorer tab, select Insert From File and select `flexb.lua`.

# Example use 

```lua
-- XOR example

local nn = require("flexb")

local ai = nn.new({2,2,1},{nn.swish,nn.linear})

local ds = {
	[{0,0}] = {0},
	[{0,1}] = {1},
	[{1,0}] = {1},
	[{1,1}] = {0}
}

local clk,epoch = os.clock(),1
while true do
	local s = 0
	for k,v in next,ds do
		local outp,trin = ai:forward(k)
		ai:backward(trin,nn.huberderiv(outp,v),0.02,0.001,0.9,0.999)
		local loss = nn.huber(outp,v)
		print(loss)
		s = s + loss
	end
	if s < 1e-9 then break end -- likely takes <1000 epochs to get there
	epoch = epoch + 1
end
print("\nTime taken:",os.clock()-clk)
print("Epochs taken:",epoch)

local c = 0
for k,v in next,ds do
	local outp = ai:forward(k)
	c = c + nn.huber(outp,v)
end

print("\nAI's total inaccuracy:",c)

print("\n#1.1.",ai.weight[1][1][1],ai.weight[1][1][2],ai.bias[1][1],ai.gamma[1][1],ai.beta[1][1])
print("#1.2.",ai.weight[1][2][1],ai.weight[1][2][2],ai.bias[1][2],ai.gamma[1][2],ai.beta[1][2])
print("#2.1.",ai.weight[2][1][1],ai.weight[2][1][2],ai.bias[2][1],ai.gamma[2][1],ai.beta[2][1])
```
