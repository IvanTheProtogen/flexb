# Installation 

You just download `flexb.lua` and place it in the directory your script will access through.

For Roblox, right-click ReplicatedFirst in the Explorer tab, select Insert From File and select `flexb.lua`.

# Example use 

```lua
local nn = require("flexb")

local mynn = nn.new({
	{
		nn.neuron.new("relu","relu",{0.7,0.3},0.1),
		nn.neuron.new("relu","relu",{1.2,0.7},0.2)
	},
	{
		nn.neuron.new("relu","relu",{0.9,1.1},0.3)
	}
})

for _=1,10000 do
	local a,b = math.random(1,10),math.random(1,10)
	local x = a+b
	local lout,lsum = mynn:forward({a,b})
	local changes = mynn:backward({a,b},lout,lsum,{x})
	mynn:update(changes)
end

local lout = mynn:forward({5,3})
print(lout[#lout][1]-8)
```
