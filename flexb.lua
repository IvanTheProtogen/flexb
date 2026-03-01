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

function nn.sigmoid(x)
	return 1/(1+math.exp(-x))
end
function nn.relu(x)
	return x>0 and x or 0
end
function nn.linear(x)
	return x
end
function nn.swish(x)
	return x*(nn.sigmoid(x))
end
nn.tanh = math.tanh or function(x)
	return (2/(1+math.exp(-2*x)))-1
end
function nn.hardsigmoid(x)
	if x < 0 then return 0 end
	if x > 1 then return 1 end
	return x
end
function nn.hardtanh(x)
	if x < -1 then return -1 end
	if x > 1 then return 1 end
	return x
end

function neuron.new(activ,weight,bias,mask)
	bias = bias or 0
	if type(weight)=="number" then
		local newweight = {}
		local scale = math.sqrt(1/weight)
		for i=1,weight do 
			if (mask and mask[1][i]) == 0 then 
				table.insert(newweight,0)
			else
				table.insert(newweight,(math.random() * 2 - 1) * scale)
			end
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
	obj.m_w = {}
	obj.v_w = {} 
	obj.m_b = 0
	obj.v_b = 0
	obj.t = 0
	obj.mean = 0
	obj.std = 1
	obj.count = 0
	obj.gamma = 1
	obj.beta = 0
	obj.m_gamma = 0
	obj.v_gamma = 0
	obj.m_beta = 0
	obj.v_beta = 0
	for i=1,#weight do
		obj.m_w[i] = 0
		obj.v_w[i] = 0
	end
	setmetatable(obj,neuron)
	return obj
end

function neuron.forward(self,input)
	assert(#input == #self.weight,"invalid input size")
	local sum = self.bias
	for i=1,#input do
		sum = sum + input[i]*self.weight[i]
	end
	local normalized = (sum - self.mean) / (self.std + 1e-8)
	local scaled = self.gamma * normalized + self.beta
	return self.activ(scaled), scaled
end

function nn.deriv(f,x)
	local e = 0.0001+(x*0.000000000000001)
	local a = f(x+e)
	local b = f(x-e)
	return (a-b)/(2*e)
end

function neuron.backward(self,output,sum,target,epsilon)
	return (target - output) * nn.deriv(self.activ, sum) * self.gamma / (self.std + (epsilon or 1e-8))
end

function neuron.update(self,input,error,power,lambda,beta1,beta2,epsilon)
	power = power or 0.001
	lambda = lambda or 0.001
	beta1 = beta1 or 0.9
	beta2 = beta2 or 0.999
	epsilon = epsilon or 1e-8
	local mask = self.mask
	self.t = self.t + 1 
	local decay = 1 - power * lambda
	if decay < epsilon then decay = epsilon end 
	local current_sum = self.bias
	for i=1,#input do
		current_sum = current_sum + input[i] * self.weight[i]
	end
	if self.count > 0 then
		self.count = self.count + 1
		local alpha = 1.0 / self.count
		self.mean = (1 - alpha) * self.mean + alpha * current_sum
		local diff = current_sum - self.mean
		self.std = math.sqrt((1 - alpha) * (self.std * self.std) + alpha * (diff * diff))
	else
		self.mean = current_sum
		self.std = 1.0
		self.count = 1
	end
	local normalized = (current_sum - self.mean) / (self.std + 1e-8)
	local norm_grad_gamma = error * normalized
	local norm_grad_beta = error
	self.m_gamma = beta1 * self.m_gamma + (1 - beta1) * norm_grad_gamma
	self.v_gamma = beta2 * self.v_gamma + (1 - beta2) * norm_grad_gamma * norm_grad_gamma
	local m_hat_gamma = self.m_gamma / (1 - beta1^self.t)
	local v_hat_gamma = self.v_gamma / (1 - beta2^self.t)
	self.gamma = self.gamma + power * m_hat_gamma / (math.sqrt(v_hat_gamma) + epsilon)
	self.m_beta = beta1 * self.m_beta + (1 - beta1) * norm_grad_beta
	self.v_beta = beta2 * self.v_beta + (1 - beta2) * norm_grad_beta * norm_grad_beta
	local m_hat_beta = self.m_beta / (1 - beta1^self.t)
	local v_hat_beta = self.v_beta / (1 - beta2^self.t)
	self.beta = self.beta + power * m_hat_beta / (math.sqrt(v_hat_beta) + epsilon)
	for i=1,#input do
		local grad = error * input[i] * (mask and mask[1][i] or 1)
		self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * grad
		self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * grad * grad
		local m_hat = self.m_w[i] / (1 - beta1^self.t)
		local v_hat = self.v_w[i] / (1 - beta2^self.t)
		self.weight[i] = self.weight[i] * decay + power * m_hat / (math.sqrt(v_hat) + epsilon)
	end
	local grad = error * (mask and mask[2] or 1)
	self.m_b = beta1 * self.m_b + (1 - beta1) * grad
	self.v_b = beta2 * self.v_b + (1 - beta2) * grad * grad
	local m_hat = self.m_b / (1 - beta1^self.t)
	local v_hat = self.v_b / (1 - beta2^self.t)
	self.bias = self.bias * decay + power * m_hat / (math.sqrt(v_hat) + epsilon)
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

function nn.backward(layers,lout,lsum,target,epsilon)
	local input = lout[1]
	local errors = {}
	local current_errors = {}
	local output_layer = layers[#layers]
	for j, neuron in ipairs(output_layer) do
		local output = lout[#lout][j]
		local sum = lsum[#lsum][j]
		table.insert(current_errors,neuron:backward(output,sum,target[j] or target,epsilon))
	end
	errors[#layers] = current_errors
	for i = #layers - 1, 1, -1 do
		current_errors = {}
		local layer = layers[i]
		local next_layer = layers[i + 1]
		for j, neuron in ipairs(layer) do
			local error_sum = 0
			for k, next_neuron in ipairs(next_layer) do
				error_sum = error_sum + errors[i + 1][k] * next_neuron.weight[j]
			end
			table.insert(current_errors, error_sum * nn.deriv(neuron.activ,lsum[i][j]) * neuron.gamma / (neuron.std + (epsilon or 1e-8)))
		end
		errors[i] = current_errors
	end
	local changes = {}
	for i, layer in ipairs(layers) do
		local layer_input = lout[i]
		for j, neuron in ipairs(layer) do
			local error = errors[i][j]
			if not changes[neuron] then
				changes[neuron] = {layer_input, error}
			end
		end
	end
	return changes
end

function nn.update(changes,power,lambda,beta1,beta2,epsilon)
	for neuron, update_data in pairs(changes) do
		neuron:update(update_data[1],update_data[2],power,lambda,beta1,beta2,epsilon)
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

function nn.loss(pred,trg)
	local loss,len = 0,#pred
	for i=1,len do
		loss = loss + (pred[i]-trg[i])^2
	end
	return loss/len
end

return nn
