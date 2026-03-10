> [!IMPORTANT]
> FlexB is still work-in-progress, please don't expect stability. Newer versions *will* break compatibility.

# Installation 

You just download `flexb.lua` and place it in the directory your script will access through.

For Roblox, right-click ReplicatedFirst in the Explorer tab, select Insert From File and select `flexb.lua`.

# Example use 

```lua
-- XOR example

local nn = require("flexb")

local ai = nn.new({
	{
		nn.neuron.new(nn.linear,{true,false}),
		nn.neuron.new(nn.relu,{false,true}),
	},
	{
		nn.neuron.new(nn.linear,{true,true})
	}
})

local ds = {
	[{0,0}] = {0},
	[{1,0}] = {1},
	[{0,1}] = {1},
	[{1,1}] = {0}
}

local clk = os.clock()
for i=1,20000 do
	for k,v in next,ds do
		local lout,lsum = ai:forward(k)
		nn.update(ai1:backward(lout,lsum,v))
	end
	print("Epochs left:",20001-i)
end
print("\nTime taken:",os.clock()-clk)

local c1,c2 = 0,0
for k,v in next,ds do
	local lout = ai:forward(k)
	c1 = c1 + nn.loss.mse(lout[#lout],v)
end

print("\nAI's total inaccuracy:",c1)

print("\n#1.1.",ai[1][1].weight[1],ai[1][1].weight[2],ai[1][1].bias)
print("#1.2.",ai[1][2].weight[1],ai[1][2].weight[2],ai[1][2].bias)
print("#2.1.",ai[2][1].weight[1],ai[2][1].weight[2],ai[2][1].bias)
```
