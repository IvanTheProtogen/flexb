-- FlexB
-- A neural network module for Lua
-- https://github.com/IvanTheProtogen/flexb

--!native
--!optimize 2

local msqrt,mrandom,mlog,mexp,tinsert = math.sqrt, math.random, math.log, math.exp, table.insert
local mmax,posinf,neginf,mabs,mmin = math.max, math.huge, -math.huge, math.abs, math.min

local nn = {}
nn.__index = nn

function nn.new(sizes,activ)
	local self = setmetatable({},nn)
	self.sizes = sizes
	self.nlayers = #sizes - 1
	self.activ = activ
	self.weight = {}
	self.bias = {}
	self.gamma = {}
	self.beta = {}
	self.drprnd = {}
	self.m_w = {}
	self.v_w = {}
	self.m_b = {}
	self.v_b = {}
	self.m_gamma = {}
	self.v_gamma = {}
	self.m_beta = {}
	self.v_beta = {}
	self.t = 0
	for i=1,self.nlayers do
		local nin = sizes[i]
		local nout = sizes[i+1]
		local scale = msqrt(8/(nin+nout))
		self.weight[i] = {}
		self.bias[i] = {}
		self.gamma[i] = {}
		self.beta[i] = {}
		self.drprnd[i] = {}
		self.m_w[i] = {}
		self.v_w[i] = {}
		self.m_b[i] = {}
		self.v_b[i] = {}
		self.m_gamma[i] = {}
		self.v_gamma[i] = {}
		self.m_beta[i] = {}
		self.v_beta[i] = {}
		for j=1,nout do
			self.weight[i][j] = {}
			self.m_w[i][j] = {}
			self.v_w[i][j] = {}
			for k=1,nin do
				self.weight[i][j][k] = (mrandom()-0.5)*scale
				self.m_w[i][j][k] = 0
				self.v_w[i][j][k] = 0
			end
			self.bias[i][j] = 0
			self.gamma[i][j] = 1
			self.beta[i][j] = 0
			self.drprnd[i][j] = 0
			self.m_b[i][j] = 0
			self.v_b[i][j] = 0
			self.m_gamma[i][j] = 0
			self.v_gamma[i][j] = 0
			self.m_beta[i][j] = 0
			self.v_beta[i][j] = 0
		end
	end
	self.adapt = {}
	self.iadapt = {}
	for i=1,self.nlayers do
		self.adapt[i] = msqrt(sizes[i])
		self.iadapt[i] = 1/self.adapt[i]
	end
	local minsiz = math.huge
	for i=1,self.nlayers-1 do
		minsiz = mmin(minsiz,sizes[i+1])
	end
	self.drpout = {}
	for i=1,self.nlayers-1 do
		self.drpout[i] = mmin(0.5,mmax(0,1 - minsiz / sizes[i+1]))
	end
	self.drpout[self.nlayers] = 0
	return self
end

function nn.forward(self,x)
	local outp = {x}
	local sums = {}
	local norm = {}
	local xhat = {}
	local rmss = {}
	for i=1,self.nlayers do
		local inp = outp[i]
		local nin = self.sizes[i]
		local nout = self.sizes[i+1]
		local lsum = {}
		local lnorm = {}
		local lout = {}
		local lxhat = {}
		local rms = 0
		for j=1,nout do
			local sum = self.bias[i][j]
			for k=1,nin do
				sum = sum + inp[k] * self.weight[i][j][k]
			end
			lsum[j] = sum
		end
		for j=1,nout do
			rms = rms + lsum[j]*lsum[j]
		end
		rms = msqrt(rms / nout + 1e-5)
		local irms = 1/rms
		for j=1,nout do
			local xhat = lsum[j] * irms
			lxhat[j] = xhat
			lnorm[j] = self.gamma[i][j] * xhat + self.beta[i][j]
			lout[j] = self.activ[i](lnorm[j])
		end
		if self.drpout[i] > 0 then
			local p = self.drpout[i]
			local scale = 1/(1-p)
			for j=1,nout do
				if self.drprnd[i][j] > p then
					lout[j] = lout[j] * scale
				else
					lout[j] = 0
				end
			end
		end
		tinsert(outp, lout)
		tinsert(sums, lsum)
		tinsert(norm, lnorm)
		tinsert(xhat,lxhat)
		tinsert(rmss,rms)
	end
	return outp[#outp], {outp, sums, norm, xhat, rmss}
end

function nn.backward(self, trin, grad, lr, lambda, beta1, beta2)
	local outp = trin[1]
	local sums = trin[2]
	local norm = trin[3]
	local xhat = trin[4]
	local rmss = trin[5]
	lr = lr or 0.004
	lambda = lambda or 0.001
	beta1 = beta1 or 0.9
	beta2 = beta2 or 0.999
	local eps = 1e-8
	self.t = self.t + 1
	local t = self.t
	local beta1_t = 1/(1-beta1^t)
	local beta2_t = 1/(1-beta2^t)
	local delta = {}
	for k,v in next,grad do
		delta[k] = v
	end
	for i=self.nlayers,1,-1 do
		local inp = outp[i]
		local nin = self.sizes[i]
		local nout = self.sizes[i+1]
		local ndiv = 1/nout
		local wlr = lr * self.adapt[i]
		local wlambda = lambda * self.iadapt[i]
		local xhat = xhat[i]
		local rms = rmss[i]
		local dnorm = {}
		local irms = 1/rms
		local p = self.drpout[i]
		local scale = p>0 and (1/(1-p)) or 1
		for j=1,nout do
			self.drprnd[i][j] = mrandom()
			local mask = self.drprnd[i][j] <= p and 0 or 1
			dnorm[j] = delta[j] * nn.deriv(self.activ[i],norm[i][j]) * mask * scale
			local grad_gamma = dnorm[j] * xhat[j]
			local grad_beta = dnorm[j]
			self.m_gamma[i][j] = beta1 * self.m_gamma[i][j] + (1 - beta1) * grad_gamma
			self.v_gamma[i][j] = beta2 * self.v_gamma[i][j] + (1 - beta2) * grad_gamma * grad_gamma
			local m_hat_g = self.m_gamma[i][j] * beta1_t
			local v_hat_g = self.v_gamma[i][j] * beta2_t
			self.gamma[i][j] = self.gamma[i][j] - wlr * m_hat_g / (msqrt(v_hat_g) + eps) - wlr * wlambda * self.gamma[i][j]
			self.m_beta[i][j] = beta1 * self.m_beta[i][j] + (1 - beta1) * grad_beta
			self.v_beta[i][j] = beta2 * self.v_beta[i][j] + (1 - beta2) * grad_beta * grad_beta
			local m_hat_beta = self.m_beta[i][j] * beta1_t
			local v_hat_beta = self.v_beta[i][j] * beta2_t
			self.beta[i][j] = self.beta[i][j] - wlr * m_hat_beta / (msqrt(v_hat_beta) + eps) - wlr * wlambda * self.beta[i][j]
		end
		local sdnxhat = 0
		for j=1,nout do
			sdnxhat = sdnxhat + dnorm[j] * xhat[j]
		end
		local div = 1/nout
		for j=1,nout do
			delta[j] = (dnorm[j] * self.gamma[i][j] - xhat[j] * (sdnxhat * div)) * irms
		end
		for j=1,nout do
			local dj = delta[j]
			self.m_b[i][j] = beta1 * self.m_b[i][j] + (1 - beta1) * dj
			self.v_b[i][j] = beta2 * self.v_b[i][j] + (1 - beta2) * dj * dj
			local m_hat_b = self.m_b[i][j] * beta1_t
			local v_hat_b = self.v_b[i][j] * beta2_t
			self.bias[i][j] = self.bias[i][j] - wlr * m_hat_b / (msqrt(v_hat_b) + eps) - wlr * wlambda * self.bias[i][j]
			for k=1,nin do
				local grad_w = dj * inp[k]
				self.m_w[i][j][k] = beta1 * self.m_w[i][j][k] + (1 - beta1) * grad_w
				self.v_w[i][j][k] = beta2 * self.v_w[i][j][k] + (1 - beta2) * grad_w * grad_w
				local m_hat_w = self.m_w[i][j][k] * beta1_t
				local v_hat_w = self.v_w[i][j][k] * beta2_t
				self.weight[i][j][k] = self.weight[i][j][k] - wlr * m_hat_w / (msqrt(v_hat_w) + eps) - wlr * wlambda * self.weight[i][j][k]
			end
		end
		if i > 1 then
			local ndelta = {}
			for k=1,nin do
				ndelta[k] = 0
				for j=1,nout do
					ndelta[k] = ndelta[k] + delta[j] * self.weight[i][j][k]
				end
			end
			delta = ndelta
		end
	end
end

function nn.deriv(f,x)
	local e = mabs(x) * 1e-8 + 1e-12
	return (f(x+e)-f(x-e))/(2*e)
end

function nn.sigmoid(x) return 1/(1+mexp(-x)) end
function nn.relu(x) return x>0 and x or 0 end
function nn.linear(x) return x end
function nn.swish(x) return x/(1+mexp(-x)) end
nn.tanh = math.tanh or function(x) return (2/(1+mexp(-2*x)))-1 end
function nn.hardsigmoid(x) return x<0 and 0 or x>1 and 1 or x end
function nn.hardtanh(x) return x<-1 and -1 or x>1 and 1 or x end

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

function nn.huber(pred,trg,delta) -- recommended for most cases
	local loss,len,delta = 0,#pred,delta or 1
	for k,v in next,pred do
		local diff = mabs(v-trg[k])
		if diff <= delta then
			loss = loss + 0.5 * diff^2
		else
			loss = loss + delta * diff - 0.5 * delta^2
		end
	end
	return loss/len
end

function nn.huberderiv(pred,trg,delta) -- recommended for most cases
	local grad,len,delta = {},#pred,delta or 1
	local div = 1/len
	for k,v in next,pred do
		local diff = v - trg[k]
		if mabs(diff) <= delta then
			grad[k] = diff * div
		else
			grad[k] = delta * (diff > 0 and 1 or -1) * div
		end
	end
	return grad
end

function nn.mse(pred,trg)
	local loss,len = 0,#pred
	for k,v in next,pred do
		loss = loss + (v-trg[k])^2
	end
	return loss/len
end

function nn.msederiv(pred,trg)
	local grad,len = {},#pred
	for k,v in next,pred do
		grad[k] = 2 * (v-trg[k]) / len
	end
	return grad
end

function nn.mae(pred,trg)
	local loss,len = 0,#pred
	for k,v in next,pred do
		loss = loss + mabs(v-trg[k])
	end
	return loss/len
end

function nn.maederiv(pred,trg)
	local grad,len = {},#pred
	local div = 1/len
	for k,v in next,pred do
		local diff = v - trg[k]
		if diff > 0 then
			grad[k] = div
		elseif diff < 0 then
			grad[k] = -div
		else
			grad[k] = 0
		end
	end
	return grad
end

function nn.bce(pred,trg,eps)
	local loss,len,eps = 0,#pred,eps or 1e-15
	for k,v in next,pred do
		local p = mmax(mmin(v,1-eps),eps)
		loss = loss + trg[k] * mlog(p) + (1 - trg[k]) * mlog(1-p)
	end
	return -loss/len
end

function nn.bcederiv(pred,trg)
	local grad,len,eps = {},#pred
	local div = 1/len
	for k,v in next,pred do
		local p = mmax(mmin(v, 1-1e-15),1e-15)
		grad[k] = ((p - trg[k]) / (mmax(p,1e-12) * mmax(1-p,1e-12))) * div
	end
	return grad
end

function nn.cce(pred,trg,eps)
	-- please make sure to apply softmax to `pred`!
	local loss,len,eps = 0,#pred,eps or 1e-15
	for k,v in next,pred do
		loss = loss + trg[k] * mlog(mmax(v,eps))
	end
	return -loss
end

function nn.ccederiv(pred, trg)
	-- pred should be raw logits! if you cant get raw logits, use `nn.logit`
	local grad,softmax = {},nn.softmax(pred)
	for k,v in next,softmax do
		grad[k] = v - trg[k]
	end
	return grad
end

function nn.quantile(pred,trg,tau)
	local loss,len,tau = 0,#pred,tau or 0.5
	for k,v in next,pred do
		local diff = trg[k] - v
		if diff >= 0 then
			loss = loss + tau * diff
		else
			loss = loss + (tau-1) * diff
		end
	end
	return loss/len
end

function nn.quantilederiv(pred, trg, tau)
	local grad,len,tau = {},#pred,tau or 0.5
	local div = 1/len
	for k,v in next,pred do
		local diff = trg[k] - v
		if diff > 0 then
			grad[k] = -tau * div
		elseif diff < 0 then
			grad[k] = (1 - tau) * div
		else
			grad[k] = 0
		end
	end
	return grad
end

local deepcopytable;function deepcopytable(src,dest)
	for k,v in next,src do
		if type(v) == "table" then
			dest[k] = deepcopytable(v,{})
		else
			dest[k] = v
		end
	end
	return dest
end
function nn.save(self)
	local d = deepcopytable(self,{},true)
	d.sizes = nil
	d.nlayers = nil
	d.activ = nil
	return d
end
function nn.load(self,data)
	for k,v in next,data do
		self[k] = v
	end
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
