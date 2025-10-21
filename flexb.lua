--!native
--!optimize 2

local neuron = {}
neuron.__index = neuron

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
function builtin.sigmoidDeriv(x)
	local s = builtin.sigmoid(x)
	return s*(1-s)
end
function builtin.reluDeriv(x)
	return x >= 0 and 1 or 0
end
function builtin.linearDeriv(x)
	return 1
end
function builtin.swishDeriv(x)
	local s = builtin.sigmoid(x)
	local d = s*(1-s)
	return s+(x*d)
end
function builtin.tanhDeriv(x)
	local t = math.tanh(x)
	return 1 - t * t
end
neuron.builtin = builtin

function neuron.new(activ,deriv,weight,bias)
	deriv = (type(deriv)=="string" and builtin[deriv.."Deriv"]) or (type(deriv)=="function" and deriv) or (type(activ)=="string" and builtin[activ.."Deriv"]) or error("expected function, got "..type(deriv),2)
	activ = (type(activ)=="string" and builtin[activ]) or (type(activ)=="function" and activ) or error("expected function, got ",2)
	assert(type(weight)=="table" or type(weight)=="number","expected table or number, got "..type(weight))
	if bias ~= nil then
		assert(type(bias)=="number","expected number, got "..type(bias))
	else
		bias = math.random()
	end
	if type(weight)=="number" then
		local newweight = {}
		for i=1,weight do
			table.insert(newweight,math.random() * math.sqrt(1/weight))
		end
		weight = newweight
	end
	local obj = {}
	obj.activ = activ
	obj.deriv = deriv
	obj.weight = weight
	obj.bias = bias
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

function neuron.backward(self,output,sum,target)
	assert(type(output)=="number","expected number, got "..type(output))
	assert(type(target)=="number","expected number, got "..type(target))
	local error = (target - output) * self.deriv(sum)
	return error
end

function neuron.update(self,input,error,power)
	power = (power==nil and 0.004) or (type(power)=="number" and power) or error("expected number, got "..type(power))
	for i=1,#input do
		self.weight[i] = self.weight[i] + power * error * input[i]
	end
	self.bias = self.bias + power * error
end

local nn = {}
nn.__index = nn
nn.neuron = neuron

function nn.new(layers)
	setmetatable(layers,nn)
	return layers
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

function nn.backward(layers,input,lout,lsum,target,power)
	power = (power==nil and 0.004) or (type(power)=="number" and power) or error("expected number, got "..type(power))
	
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
			local derivative = neuron.deriv(sum)
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
				local weight_update = power * error * layer_input[k]
				table.insert(changes[neuron][1], weight_update)
			end
			changes[neuron][2] = power * error
		end
	end
	
	return changes
end

function nn.update(layers,changes)
	for neuron, update_data in pairs(changes) do
		local weight_updates, bias_update = update_data[1], update_data[2]
		for i = 1, #neuron.weight do
			neuron.weight[i] = neuron.weight[i] + weight_updates[i]
		end
		neuron.bias = neuron.bias + bias_update
	end
end

return nn
