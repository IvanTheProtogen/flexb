-- FlexB
-- A neural network module for Lua
-- https://github.com/IvanTheProtogen/flexb

--!native
--!optimize 2

local nn = {}
nn.__index = nn

local neuron = {}
neuron.__index = neuron
nn.neuron = neuron

local builtin = {}
function builtin.sigmoid(x)
	return 1/(1+math.exp(-x))
end
function builtin.relu(x)
	return math.max(0,x)
end
function builtin.linear(x)
	return x
end
function builtin.swish(x)
	return x*(builtin.sigmoid(x))
end
builtin.tanh = math.tanh
nn.builtin = builtin

function neuron.new(activ,weight,bias,mask)
	activ = (type(activ)=="string" and builtin[activ]) or (type(activ)=="function" and activ) or error("expected function, got "..type(activ),2)
	assert(type(weight)=="table" or type(weight)=="number","expected table or number, got "..type(weight))
	if bias ~= nil then
		assert(type(bias)=="number","expected number, got "..type(bias))
	else
		bias = 0
	end
	if mask ~= nil then 
		assert(type(mask)=="table","expected table, got "..type(mask))
		assert(type(mask[1])=="table","expected table, got "..type(mask[1]))
		assert(type(mask[2])=="number","expected number, got "..type(mask[2]))
	end
	if type(weight)=="number" then
		local newweight = {}
		local scale = math.sqrt(1/weight)
		for i=1,weight do 
			table.insert(newweight,(math.random() * 2 - 1) * scale)
		end
		weight = newweight
	else 
		local scale = math.sqrt(1/#weight)
		for k,v in next,weight do 
			if v==true then 
				weight[k] = math.random() * scale
			elseif v==false then 
				weight[k] = -math.random() * scale
			elseif v=="" then 
				weight[k] = (math.random() * 2 - 1) * scale
			end
		end
	end
	local obj = {}
	obj.activ = activ
	obj.weight = weight
	obj.bias = bias
	obj.mask = mask
	obj.velw = {}
	for i=1,#weight do
		obj.velw[i] = 0
	end
	obj.velb = 0
	setmetatable(obj,neuron)
	return obj
end

function neuron.forward(self,input)
	assert(type(input)=="table","expected table, got "..type(input))
	assert(#input == #self.weight,"invalid input size")
	local sum = self.bias
	for i=1,#input do
		sum = sum + input[i]*self.weight[i]
	end
	return self.activ(sum),sum
end

function nn.deriv(f,x)
	local e = 0.0001+(x*0.000000000000001)
	local a = f(x+e)
	local b = f(x-e)
	return (a-b)/(2*e)
end

function neuron.backward(self,output,sum,target)
	assert(type(output)=="number","expected number, got "..type(output))
	assert(type(target)=="number","expected number, got "..type(target))
	local error = (target - output) * nn.deriv(self.activ,sum)
	return error
end

function neuron.update(self,input,error,power,momentum,ignoremask)
	power,momentum = power or 0.004, momentum or 0.9
	local mask,old_velw,old_velb = self.mask,{},self.velb
	for i=1,#input do
		if (mask and not ignoremask) and mask[1][i] ~= 0 then 
			old_velw[i] = self.velw[i]
			self.velw[i] = momentum * self.velw[i] + power * error * input[i] * mask[1][i]
			self.weight[i] = self.weight[i] - momentum * old_velw[i] + (1 + momentum) * self.velw[i]
		end
	end
	if (mask and not ignoremask) and mask[2] ~= 0 then
		self.velb = momentum * self.velb + power * error
		self.bias = self.bias - momentum * old_velb + (1 + momentum) * self.velb
	end
end

function nn.new(layers)
	setmetatable(layers,nn)
	return layers
end

function nn.layer(activ,w,c)
	local t = {}
	for i=1,c do
		table.insert(t,neuron.new(activ,w))
	end
	return t
end

function nn.forward(layers,input)
	local outputs,sums = {input},{}
	for i = 1, #layers do
		local layer_output = {}
		local layer_sums = {}
		for j, neuron in ipairs(layers[i]) do
			local output, sum = neuron:forward(outputs[i])
			table.insert(layer_output, output)
			table.insert(layer_sums, sum)
		end
		table.insert(outputs, layer_output)
		table.insert(sums, layer_sums)
	end
	return outputs, sums
end

function nn.backward(layers,lout,lsum,target)
	local input = lout[1]
	local errors = {}
	local current_errors = {}
	
	local output_layer = layers[#layers]
	for j, neuron in ipairs(output_layer) do
		local output = lout[#lout][j]
		local sum = lsum[#lsum][j]
		local error = neuron:backward(output, sum, target[j] or target)
		table.insert(current_errors, error)
	end
	errors[#layers] = current_errors
	
	for i = #layers - 1, 1, -1 do
		current_errors = {}
		local layer = layers[i]
		local next_layer = layers[i + 1]
		
		for j, neuron in ipairs(layer) do
			local sum = lsum[i][j]
			local derivative = nn.deriv(neuron.activ,sum)
			local error_sum = 0
			for k, next_neuron in ipairs(next_layer) do
				error_sum = error_sum + errors[i + 1][k] * next_neuron.weight[j]
			end
			
			local error = error_sum * derivative
			table.insert(current_errors, error)
		end
		errors[i] = current_errors
	end
	
	local changes = {}
	
	for i, layer in ipairs(layers) do
		local layer_input = lout[i]
		for j, neuron in ipairs(layer) do
			local error = errors[i][j]
			if not changes[neuron] then
				changes[neuron] = {{}, 0}
			end
			for k = 1, #layer_input do
				local weight_update = error * layer_input[k]
				table.insert(changes[neuron][1], weight_update)
			end
			changes[neuron][2] = error
		end
	end
	
	return changes
end

function nn.update(layers,changes,power,momentum,ignoremask)
	power,momentum = power or 0.004, momentum or 0.9
	for neuron, update_data in pairs(changes) do
		local weight_updates, bias_update = update_data[1], update_data[2]
		local mask,old_velw,old_velb = neuron.mask,{},neuron.velb
		for i=1,#neuron.weight do
			if (mask and not ignoremask) and mask[1][i] ~= 0 then
				old_velw[i] = neuron.velw[i]
				neuron.velw[i] = momentum * neuron.velw[i] + power * (weight_updates[i] or 0)
				neuron.weight[i] = neuron.weight[i] - momentum * old_velw[i] + (1 + momentum) * neuron.velw[i]
			end
		end
		if (mask and not ignoremask) and mask[2] ~= 0 then
			neuron.velb = momentum * neuron.velb + power * bias_update
			neuron.bias = neuron.bias - momentum * old_velb + (1 + momentum) * neuron.velb
		end
	end
end

function nn.softmax(t)
	local r,sum,max = {},0,-math.huge
	for k,v in next,t do
		max = max < v and v or max
	end
	for k,v in next,t do
		local v = math.exp(v - max)
		r[k] = v
		sum = sum + v
	end
	for k,v in next,r do
		r[k] = v/sum
	end
	return r,sum
end

function nn.logit(t,sum)
	local r,sum = {},sum or 1
	for k,v in next,t do
		r[k] = math.log(math.max(v*sum,1e-10))
	end
	return r
end

return nn
