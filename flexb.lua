-- FlexB
-- A neural network module for Lua
-- https://github.com/IvanTheProtogen/flexb

--!native
--!optimize 2

local msqrt,mrandom,mlog,mexp,tinsert = math.sqrt, math.random, math.log, math.exp, table.insert
local mmax,neginf = math.max, -math.huge

local nn = {}
nn.__index = nn

local neuron = {}
neuron.__index = neuron
nn.neuron = neuron

function nn.sigmoid(x)
	return 1/(1+mexp(-x))
end
function nn.relu(x)
	return x>0 and x or 0
end
function nn.linear(x)
	return x
end
function nn.swish(x)
	return x/(1+mexp(-x))
end
nn.tanh = math.tanh or function(x)
	return (2/(1+mexp(-2*x)))-1
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

local function neuronnew(activ,weight,bias,mask)
	bias = bias or 0
	if type(weight)=="number" then
		local newweight = {}
		local scale = msqrt(4/weight)
		for i=1,weight do 
			if (mask and mask[1][i]) == 0 then 
				tinsert(newweight,0)
			else
				tinsert(newweight,(mrandom() - 0.5) * scale)
			end
		end
		weight = newweight
	else 
		local scale = msqrt(4/#weight)
		local scale2 = scale*0.5
		for k,v in next,weight do 
			if v==true then 
				weight[k] = mrandom() * scale2
			elseif v==false then 
				weight[k] = -mrandom() * scale2
			elseif v=="" then 
				weight[k] = (mrandom() - 0.5) * scale
			end
		end
	end
	local obj = {}
	obj.activ = activ
	obj.weight = weight
	obj.adapt = msqrt(#weight)
	obj.adaptdiv = 1/obj.adapt
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
neuron.new = neuronnew

function neuron.forward(self,input,epsilon)
	assert(#input == #self.weight,"invalid input size")
	local sum = self.bias
	for i,inp in next,input do
		sum = sum + inp*self.weight[i]
	end
	local normalized = (sum - self.mean) / (self.std + (epsilon or 1e-8))
	local scaled = self.gamma * normalized + self.beta
	return self.activ(scaled), scaled
end

local function nnderiv(f,x)
	local e = 0.0001+(x*0.000000000000001)
	local a = f(x+e)
	local b = f(x-e)
	return (a-b)/(2*e)
end
nn.deriv = nnderiv

function neuron.backward(self,output,sum,target,epsilon)
	return (target - output) * nnderiv(self.activ, sum) * self.gamma / (self.std + (epsilon or 1e-8))
end

function neuron.update(self,input,error,sum,power,lambda,beta1,beta2,timestep,epsilon)
	local adapt = self.adapt
	local adaptdiv = self.adaptdiv
	power = (power or 0.001) * adapt
	lambda = (lambda or 0.01) * adaptdiv
	beta1 = (beta1 or 0.9) - 0.1 * adaptdiv
	beta2 = (beta2 or 0.999) - 0.001 * adaptdiv
	epsilon = epsilon or 1e-5
	if not timestep then
		local _temp = self.t + 1
		self.t = _temp
		timestep = _temp
	end
	local mask = self.mask
	local decay = 1 - power * lambda
	if decay < epsilon then decay = epsilon end 
	if self.count > 0 then
		self.count = self.count + 1
		local alpha = 1.0 / self.count
		self.mean = (1 - alpha) * self.mean + alpha * sum
		local diff = sum - self.mean
		self.std = msqrt((1 - alpha) * (self.std * self.std) + alpha * (diff * diff))
	else
		self.mean = sum
		self.std = 1.0
		self.count = 1
	end
	local OneMinBeta1 = 1-beta1
	local OneMinBeta2 = 1-beta2
	local Beta1Powered = 1/(1 - beta1^timestep)
	local Beta2Powered = 1/(1 - beta2^timestep)
	local normalized = (sum - self.mean) / (self.std + epsilon)
	local norm_grad_gamma = error * normalized
	local norm_grad_beta = error
	self.m_gamma = beta1 * self.m_gamma + OneMinBeta1 * norm_grad_gamma
	self.v_gamma = beta2 * self.v_gamma + OneMinBeta2 * norm_grad_gamma * norm_grad_gamma
	local m_hat_gamma = self.m_gamma * Beta1Powered
	local v_hat_gamma = self.v_gamma * Beta2Powered
	self.gamma = self.gamma + power * m_hat_gamma / (msqrt(v_hat_gamma) + epsilon)
	self.m_beta = beta1 * self.m_beta + OneMinBeta1 * norm_grad_beta
	self.v_beta = beta2 * self.v_beta + OneMinBeta2 * norm_grad_beta * norm_grad_beta
	local m_hat_beta = self.m_beta * Beta1Powered
	local v_hat_beta = self.v_beta * Beta2Powered
	self.beta = self.beta + power * m_hat_beta / (msqrt(v_hat_beta) + epsilon)
	for i,inp in next,input do
		local grad = error * inp * (mask and mask[1][i] or 1)
		self.m_w[i] = beta1 * self.m_w[i] + OneMinBeta1 * grad
		self.v_w[i] = beta2 * self.v_w[i] + OneMinBeta2 * grad * grad
		local m_hat = self.m_w[i] * Beta1Powered
		local v_hat = self.v_w[i] * Beta2Powered
		self.weight[i] = self.weight[i] * decay + power * m_hat / (msqrt(v_hat) + epsilon)
	end
	local grad = error * (mask and mask[2] or 1)
	self.m_b = beta1 * self.m_b + OneMinBeta1 * grad
	self.v_b = beta2 * self.v_b + OneMinBeta2 * grad * grad
	local m_hat = self.m_b * Beta1Powered
	local v_hat = self.v_b * Beta2Powered
	self.bias = self.bias * decay + power * m_hat / (msqrt(v_hat) + epsilon)
end

function nn.new(layers)
	setmetatable(layers,nn)
	return layers
end

function nn.layer(activ,w,c)
	local t = {}
	for i=1,c do
		tinsert(t,neuronnew(activ,w))
	end
	return t
end

function nn.forward(layers,input)
	local outputs,sums = {input},{}
	for i,layer in next,layers do
		local layer_output = {}
		local layer_sums = {}
		for j, neuron in next,layer do
			local output, sum = neuron:forward(outputs[i])
			tinsert(layer_output, output)
			tinsert(layer_sums, sum)
		end
		tinsert(outputs, layer_output)
		tinsert(sums, layer_sums)
	end
	return outputs, sums
end

function nn.backward(layers,lout,lsum,target,epsilon)
	local input = lout[1]
	local errors,sums = {},{}
	local current_errors,current_sums = {},{}
	local output_layer = layers[#layers]
	local outidx,sumidx = #lout,#lsum
	for j, neuron in next,output_layer do
		local output = lout[outidx][j]
		local sum = lsum[sumidx][j]
		tinsert(current_errors,neuron:backward(output,sum,target[j] or target,epsilon))
		tinsert(current_sums,sum)
	end
	errors[#layers] = current_errors
	sums[#layers] = current_sums
	for i = #layers - 1, 1, -1 do
		current_errors = {}
		current_sums = {}
		local layer = layers[i]
		local next_layer = layers[i + 1]
		local layer_errors = errors[i + 1]
		local layer_sums = lsum[i]
		for j, neuron in next,layer do
			local error_sum = 0
			for k, next_neuron in next,next_layer do
				error_sum = error_sum + layer_errors[k] * next_neuron.weight[j]
			end
			tinsert(current_errors, error_sum * nnderiv(neuron.activ,lsum[i][j]) * neuron.gamma / (neuron.std + (epsilon or 1e-8)))
			tinsert(current_sums,layer_sums[j])
		end
		errors[i] = current_errors
		sums[i] = current_sums
	end
	local changes = {}
	for i, layer in next,layers do
		local layer_input = lout[i]
		for j, neuron in next,layer do
			if not changes[neuron] then
				changes[neuron] = {layer_input, errors[i][j], sums[i][j]}
			end
		end
	end
	return changes
end

function nn.update(changes,power,lambda,beta1,beta2,epsilon)
	for neuron, update_data in pairs(changes) do
		neuron:update(update_data[1],update_data[2],update_data[3],power,lambda,beta1,beta2,epsilon)
	end
end

function nn.softmax(t)
	local r,sum,max = {},0
	for k,v in next,t do
		max = max < v and v or neginf
	end
	for k,v in next,t do
		local v = mexp(v - max)
		r[k] = v
		sum = sum + v
	end
	local div = 1/sum
	for k,v in next,r do
		r[k] = v*div
	end
	return r,sum
end

function nn.logit(t,sum,epsilon)
	local r,sum,epsilon = {},sum or 1,epsilon or 1e-10
	for k,v in next,t do
		r[k] = mlog(mmax(v*sum,epsilon))
	end
	return r
end

function nn.argmax(t)
	local idx,max = 0,neginf
	for i,v in next,t do
		if v > max then
			idx,max = i,v
		end
	end
	return idx,max
end

function nn.loss(pred,trg)
	local loss,len = 0,#pred
	for i=1,len do
		loss = loss + (pred[i]-trg[i])^2
	end
	return loss/len
end

return nn

--[[

MIT License

Copyright (c) 2026 IvanZaSilly

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

]]
