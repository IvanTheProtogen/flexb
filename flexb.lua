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

function neuron.new(activ,weight,bias)
	activ = (type(activ)=="string" and builtin[activ]) or (type(activ)=="function" and activ) or error("expected function, got ",2)
	assert(type(weight)=="table" or type(weight)=="number","expected table or number, got "..type(weight))
	if bias ~= nil then
		assert(type(bias)=="number","expected number, got "..type(bias))
	else
		bias = 0
	end
	if type(weight)=="number" then
		local newweight = {}
		local scale = math.sqrt(1/weight)
		for i=1,weight do 
			table.insert(newweight,(math.random() * 2 - 1) * scale)
		end
		weight = newweight
	end
	local obj = {}
	obj.activ = activ
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

function neuron.update(self,input,error,power)
	power = (power==nil and 0.004) or (type(power)=="number" and power) or error("expected number, got "..type(power))
	for i=1,#input do
		self.weight[i] = self.weight[i] + power * error * input[i]
	end
	self.bias = self.bias + power * error
end

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

function nn.backward(layers,input,lout,lsum,target)
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

function nn.update(layers,changes,power)
	power = (power==nil and 0.004) or (type(power)=="number" and power) or error("expected number, got "..type(power))
	for neuron, update_data in pairs(changes) do
		local weight_updates, bias_update = update_data[1], update_data[2]
		for i = 1, #neuron.weight do
			neuron.weight[i] = neuron.weight[i] + (weight_updates[i] * power)
		end
		neuron.bias = neuron.bias + (bias_update * power)
	end
end

return nn
