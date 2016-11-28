--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'

local M = {}
local Trainer = torch.class('USOF.Trainer', M)

function Trainer:__init(model, photometric_loss, smoothness_loss, criterion, opt, optimState)
   self.model           = model
   self.photometric_loss= photometric_loss
   self.smoothness_loss = smoothness_loss
   self.criterion       = criterion
   self.optimState      = optimState or {
      learningRate      = opt.learningRate,
      learningRateDecay = 0.0,
      beta1             = opt.beta_1,
      beta2             = opt.beta_2,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)
   local timer              = torch.Timer()
   local dataTimer          = torch.Timer()
   local criterion_output   = 0.0

   local function feval()
    --   return self.criterion.output, self.gradParams
      return criterion_output, self.gradParams
   end

   local trainSize  = dataloader:size()
   local lossSum    = 0.0
   local N          = 0
   local debug_loss = 0.0

   print('=============================')
   print(self.optimState)
   print('=============================')
   
   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)
      
    --    image.save('tmp_' .. tostring(n) .. '.png',self.input[{ {},{1,3},{},{} }]:reshape(3,self.input:size(3),self.input:size(4)))

      local output              = self.model:forward(self.input):float()
      local batchSize           = output:size(1)
      
      self.photometric_loss:forward{self.input[{ {},{4,6},{},{} }] , self.model.output}
	  
	  --------------------------------------------------------------------------------
	  local grids = nn.AffineGridGeneratorUSOF():forward(output:cuda()):transpose(3,4):transpose(2,3)
	  local idx   = (grids:gt(1) + grids:lt(-1)):sum(2):ge(1):repeatTensor(1,3,1,1)
	  --------------------------------------------------------------------------------

	  local warped_img_masked   = self.photometric_loss.output:clone()
	  local ref_img_masked      = self.input[{ {},{1,3},{},{} }]:clone()

--	  local photo_loss          = self.criterion:forward(self.photometric_loss.output , self.input[{ {},{1,3},{},{} }])
--      local smooth_loss         = self.opt.smooth_weight * self.smoothness_loss:forward(self.model.output , nil)
	  
      local photo_loss			= self.criterion:forward(warped_img_masked:maskedFill(idx, 0) , ref_img_masked:maskedFill(idx, 0))
	  local smooth_loss			= self.opt.smooth_weight * self.smoothness_loss:forward(self.model.output , nil)


      criterion_output = photo_loss + smooth_loss
      debug_loss = debug_loss + photo_loss + smooth_loss

      local total_gradInput
      self.model:zeroGradParameters()


--      self.criterion:backward(self.photometric_loss.output , self.input[{ {},{1,3},{},{} }])
      self.criterion:backward(warped_img_masked:maskedFill(idx, 0) , ref_img_masked:maskedFill(idx, 0))    
     
      local photo_grads     = self.photometric_loss:backward({self.input[{ {},{4,6},{},{} }] , self.model.output} , self.criterion.gradInput)
      local smooth_grads    = self.opt.smooth_weight * self.smoothness_loss:backward(self.model.output , nil)

      total_gradInput       = (photo_grads[2]:transpose(3,4):transpose(2,3) + smooth_grads)


      self.model:backward(self.input, total_gradInput)

      local _, tmp_loss = optim.adam(feval, self.params, self.optimState)

      lossSum = lossSum + criterion_output
      N = n

      if (n%100) == 0 then
     
        --  gnuplot.pngfigure('traininLoss_' .. tostring(epoch) .. '.png')
        --  gnuplot.plot({ torch.range(1, #losses), torch.Tensor(losses), '-' })
        --  gnuplot.plotflush()
        --   debug_loss = 0.0

          print(string.format('Gradient min: %1.4f \t max:  %1.4f \t norm: %1.4f', torch.min(self.gradParams:float()), torch.max(self.gradParams:float()), torch.norm(self.gradParams:float())))
          print(string.format('output u_min: %1.4f \t u_max:%1.4f',torch.min(output[{ {1},{1},{},{} }]),torch.max(output[{ {1},{1},{},{} }])))
          print(string.format('output v_min: %1.4f \t v_max:%1.4f',torch.min(output[{ {1},{2},{},{} }]),torch.max(output[{ {1},{2},{},{} }])))

          image.save('losses/current_warped_img.png', self.photometric_loss.output[{ {1},{},{},{} }]:reshape(3,self.photometric_loss.output:size(3),self.photometric_loss.output:size(4)))
		  
		  local tmpOut = torch.zeros(1, 2, output:size(3), output:size(4))
		  tmpOut[{ {1},{1},{},{} }] = output[{ {1},{2},{},{} }]:clone()
	      tmpOut[{ {1},{2},{},{} }] = output[{ {1},{1},{},{} }]:clone()
          local tmpIMG = uvToColor(tmpOut[{ {1},{},{},{} }]:reshape(2,output:size(3),output:size(4))):div(255)
		  local gtIMG  = uvToColor(self.gt_flow[{ {1},{1,2},{},{} }]:float():reshape(2, self.gt_flow:size(3), self.gt_flow:size(4))):div(255)
   	      
		  image.save('losses/gt_flow.png', gtIMG)
		  image.save('losses/current_flow.png', tmpIMG)
          image.save('losses/current_ref.png', self.input[{ {1},{1,3},{},{} }]:reshape(3,self.input[{ {1},{1,3},{},{} }]:size(3),self.input[{ {1},{1,3},{},{} }]:size(4)))
          image.save('losses/current_tar.png', self.input[{ {1},{4,6},{},{} }]:reshape(3,self.input[{ {1},{4,6},{},{} }]:size(3),self.input[{ {1},{4,6},{},{} }]:size(4)))
          -- image.save('current_ref.png', self.input[{ {1},{1},{},{} }]:reshape(1,self.input[{ {1},{1},{},{} }]:size(3),self.input[{ {1},{1},{},{} }]:size(4)))
          -- image.save('current_tar.png', self.input[{ {1},{2},{},{} }]:reshape(1,self.input[{ {1},{2},{},{} }]:size(3),self.input[{ {1},{2},{},{} }]:size(4)))


      end
	if (n%10) == 0 then
   	     print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  photoLoss %1.4f  smoothLoss %1.4f  loss %1.4f'):format(
             epoch, n, trainSize, timer:time().real, dataTime, photo_loss, smooth_loss, criterion_output))--total_loss))
   	   -- check that the storage didn't get changed due to an unfortunate getParameters call
   	end

 	assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end

--    return top1Sum / N, top5Sum / N, lossSum / N
    return lossSum / N
end

function Trainer:test(epoch, dataloader)

   local timer = torch.Timer()
   local size = dataloader:size()
   local avgEPE, errPixels  = 0.0, 0.0
   local N                  = 0
   local criterion_output   = 0.0
   local lossSum			= 0.0

   self.model:evaluate()
   for n, sample in dataloader:run() do
      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output              = self.model:forward(self.input):float()
      local batchSize           = output:size(1)

      self.photometric_loss:forward{self.input[{ {},{4,6},{},{} }] , self.model.output}
	  --------------------------------------------------------------------------------
	  local grids = nn.AffineGridGeneratorUSOF():forward(output:cuda()):transpose(3,4):transpose(2,3)
	  local idx   = (grids:gt(1) + grids:lt(-1)):sum(2):ge(1):repeatTensor(1,3,1,1)
	  --------------------------------------------------------------------------------

	  local warped_img_masked   = self.photometric_loss.output:clone()
	  local ref_img_masked      = self.input[{ {},{1,3},{},{} }]:clone()

--	  local photo_loss          = self.criterion:forward(self.photometric_loss.output , self.input[{ {},{1,3},{},{} }])
--      local smooth_loss         = self.opt.smooth_weight * self.smoothness_loss:forward(self.model.output , nil)
	  
      local photo_loss			= self.criterion:forward(warped_img_masked:maskedFill(idx, 0) , ref_img_masked:maskedFill(idx, 0))
	  local smooth_loss			= self.opt.smooth_weight * self.smoothness_loss:forward(self.model.output , nil)


      criterion_output = photo_loss + smooth_loss
	  lossSum = lossSum + criterion_output
      N = n


   	  local average_epe, erroneous_pixels = evaluateEPE(self.model.output, self.gt_flow, 3)

      if (n%10) then    
	  	  local tmpOut = torch.zeros(1, 2, output:size(3), output:size(4))
		  tmpOut[{ {1},{1},{},{} }] = output[{ {1},{2},{},{} }]:clone()
	      tmpOut[{ {1},{2},{},{} }] = output[{ {1},{1},{},{} }]:clone()
          local tmpIMG = uvToColor(tmpOut:reshape(2,output:size(3),output:size(4))):div(255)
		  local gtIMG  = uvToColor(self.gt_flow[{ {1},{1,2},{},{} }]:float():reshape(2, self.gt_flow:size(3), self.gt_flow:size(4))):div(255)
   	      
		  image.save('losses/testing/warped_img.png', self.photometric_loss.output[{ {1},{},{},{} }]:reshape(3,self.photometric_loss.output:size(3),self.photometric_loss.output:size(4)))

		  image.save('losses/testing/gt_flow.png', gtIMG)
    	  image.save('losses/testing/test_flow.png', tmpIMG)
          image.save('losses/testing/test_ref.png', self.input[{ {1},{1,3},{},{} }]:reshape(3,self.input[{ {1},{1,3},{},{} }]:size(3),self.input[{ {1},{1,3},{},{} }]:size(4)))
          image.save('losses/testing/test_tar.png', self.input[{ {1},{4,6},{},{} }]:reshape(3,self.input[{ {1},{4,6},{},{} }]:size(3),self.input[{ {1},{4,6},{},{} }]:size(4)))
      end
          print((' | Test: [%d][%d/%d]    Time %.3f  loss %1.4f Average EPE %1.3f erroneous_pixels %1.3f'):format( epoch, n, size, timer:time().real, criterion_output, average_epe, erroneous_pixels))
          -- print((' | Test: [%d][%d/%d]    Time %.3f  loss %1.4f   fEPE_below[5pxl] %1.3f  EPE_above[5pxl] %1.3f EPE_all %1.3f erroneous_pixels %1.3f'):format( epoch, n, size, timer:time().real, photo_loss, epe_under, epe_over, epe_all, erroneous_pixels))
          -- print((' | Test: [%d][%d/%d]    Time %.3f  loss %1.4f'):format( epoch, n, size, timer:time().real, photo_loss + smooth_loss))

      avgEPE = avgEPE + average_epe
      errPixels= errPixels + erroneous_pixels

      timer:reset()
   end
   self.model:training()

  --  return EPE_below_total/N, EPE_above_total/N, EPE_all_total/N, criterion_output/N
   return lossSum/N, avgEPE/N, errPixels/N
   end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.gt_flow = self.gt_flow or torch.CudaTensor()

   self.input:resize(sample.input:size()):copy(sample.input)
   self.gt_flow:resize(sample.gt_flow:size()):copy(sample.gt_flow)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   if (self.opt.dataset == 'FlyingChairs') and (epoch >= 120) then
      if (epoch%30 == 0) then
      	return self.optimState.learningRate/2
      else
        return self.optimState.learningRate
      end
   else
	    return self.optimState.learningRate
   end 
end

return M.Trainer
