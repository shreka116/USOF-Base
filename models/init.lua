--
--  Copyright (c) 2016, Computer Vision Lab @ Seoul National University.
--  All rights reserved.
--
--
--  Model creating code

require 'nn'
require 'cunn'
require 'cudnn'
require 'tvnorm-nn'
require 'stn'
require 'nngraph'

-- require 'BrightnessCriterion'
require '../AffineGridGeneratorUSOF'

local M = {}

function M.setup(opt, checkpoint)
    local model

    if checkpoint then
        local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
        assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
        print('=> Resuming model from ' .. modelPath)
        model   = torch.load(modelPath):cuda()
    elseif opt.retrain ~= 'none' then
        assert(paths.filep(modelPath), 'Model not found: ' .. opt.retrain)
        print('=> Loading model from ' .. opt.retrain)
        model   = torch.load(opt.retrain):cuda()
    else
        print('=> Creating model from: models/' .. opt.networkType .. '.lua')
        model = require('models/' .. opt.networkType)(opt)
    end

   -- Set the CUDNN flags
   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      -- Use a deterministic convolution implementation
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end    

   -- Wrap the model with DataParallelTable, if using more than one GPU
   if opt.nGPU > 1 then
      local gpus = torch.range(1, opt.nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
   end

   local imgH   = 0
   local imgW   = 0

   if opt.dataset == "FlyingChairs" then
      imgH = 384
      imgW = 512
   elseif opt.dataset == "mpiSintel" then
      imgH = 436
      imgW = 1024
   elseif opt.dataset == "KITTI2012" then
      imgH = 384
      imgW = 512
   elseif opt.dataset == "UCF101" then
      imgH = 240
      imgW = 320
   end


   -- Define phtometric Loss
--    local photometric = nn.BrightnessCriterion():cuda()

   local epsilon             = opt.epsilon -- 0.001
   local char_power          = opt.photo_char
   local photometric         = nn.Sequential()
   local parallel_2          = nn.ParallelTable()
   local target_sequential   = nn.Sequential()
   local parallel_1          = nn.ParallelTable()
   local trans               = nn.Sequential()
   trans:add(nn.Identity())
   trans:add(nn.Transpose({2,3},{3,4}))
   parallel_1:add(trans)
   parallel_1:add(nn.AffineGridGeneratorUSOF())

   target_sequential:add(parallel_1)
   target_sequential:add(nn.BilinearSamplerBHWD())
   target_sequential:add(nn.Transpose({3,4},{2,3}))

--    parallel_2:add(nn.Identity())
--    parallel_2:add(target_sequential)
   
--    photometric:add(parallel_2)
--    photometric:add(nn.CSubTable())
--    photometric:add(nn.Square())
--    photometric:add(nn.AddConstant(epsilon))
--    photometric:add(nn.Power(char_power))


   -- Define smoothness constraint
   local smoothness = nn.Sequential()
   smoothness:add(nn.SpatialTVNormCriterion())
--    smoothness:add(nn.MulConstant(opt.smooth_weight, true))

--    local criterion  = nn.MSECriterion()
   local criterion  = nn.SmoothL1Criterion()

   model:cuda()
   target_sequential:cuda()
--    photometric:cuda()
   smoothness:cuda()
   criterion:cuda()

--    cudnn.convert(photometric, cudnn)
   cudnn.convert(target_sequential, cudnn)
   cudnn.convert(model, cudnn)

--    return model, photometric, smoothness, criterion
   return model, target_sequential, smoothness, criterion

end

return M
