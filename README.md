> [!IMPORTANT]
> FlexB is still work-in-progress, please don't expect stability. Newer versions *will* break compatibility.

# Installation 

You just download `flexb.lua` and place it in the directory your script will access through.

For Roblox, right-click ReplicatedFirst in the Explorer tab, select Insert From File and select `flexb.lua`.

# Example use 

```lua
-- XOR example

local nn = require("flexb")

local function createAI()
	return nn.new({
		{
			nn.neuron.new(nn.linear,{true,false}),
			nn.neuron.new(nn.relu,{false,true}),
		},
		{
			nn.neuron.new(nn.linear,{true,true})
		}
	})
end
local ai1,ai2 = createAI(),createAI()

local ds = {
	[{0,0}] = {0},
	[{1,0}] = {1},
	[{0,1}] = {1},
	[{1,1}] = {0}
}

for i=1,20000 do
	for k,v in next,ds do
		local lout1,lsum1 = ai1:forward(k)
		local lout2,lsum2 = ai2:forward(k)
		nn.update(ai1:backward(lout1,lsum1,v),0.01,0.001)
		nn.update(ai2:backward(lout2,lsum2,v),0.015,0.001)
	end
	print("Epochs left:",20001-i)
end

local c1,c2 = 0,0
for k,v in next,ds do
	local lout1 = ai1:forward(k)
	local lout2 = ai2:forward(k)
	c1 = c1 + nn.loss(lout1[#lout1],v)
	c2 = c2 + nn.loss(lout2[#lout2],v)
end

print("\nAI #1's total inaccuracy:",c1)
print("AI #2's total inaccuracy:",c2)
if c1 < c2 then
	print("\nAI #1 is more accurate than AI #2.")
elseif c1 > c2 then
	print("\nAI #2 is more accurate than AI #1.")
else
	print("\nAI #1 is as accurate as AI #2.")
end
```
