--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Image transforms for data augmentation and input normalization
--

require 'image'
require 'nn'

local M = {}

function M.SelectTransform(transforms)
   return function(input)
      local trTypes     = #transforms-- + 1           -- added one for identity (w/o transformation)
      local photometric = torch.random(1,6)
      local geometric   = torch.random(6,10)

      local paddingLayer= nn.SpatialReflectionPadding(math.ceil(input:size(3)*0.2)):float()
      
      local pad_input = paddingLayer:forward(input)

      if trTypes ~= 1 then
        pad_input  = transforms[photometric](pad_input)
        pad_input  = transforms[geometric](pad_input)

        local rel_trans = torch.random(11,13)
        local rel_bool  = torch.random(1,3)
        if rel_bool == 1 then
          input[{ {1,3},{},{} }]  = transforms[rel_trans](pad_input[{ {1,3},{},{} }])
          input[{ {4,6},{},{} }]  = transforms[trTypes](pad_input[{ {4,6},{},{} }])
        elseif rel_bool then 
          input[{ {1,3},{},{} }]  = transforms[trTypes](pad_input[{ {1,3},{},{} }])
          input[{ {4,6},{},{} }]  = transforms[rel_trans](pad_input[{ {4,6},{},{} }])
         else
          input  = transforms[trTypes](pad_input)
        end      
      end

      return input
   end
end

function M.Compose(transforms)
   return function(input)
      for idx, transform in ipairs(transforms) do
        --  print(tostring(idx) .. '-->' .. tostring(transform))
         input = transform(input)
        --  print(tostring(idx) .. '-input -->' .. tostring(input:size()))
      end
    --   print('outa for loop-input -->' .. tostring(input:size()))
      return input
   end
end

function M.ColorNormalize(meanstd)
   return function(img)
      img = img:clone()
      for i=1,3 do
         img[i]:add(-meanstd.mean[i])
         img[i]:div(meanstd.std[i])
      end
      return img
   end
end


-- Lighting noise (AlexNet-style PCA-based noise)
function M.Lighting(alphastd, eigval, eigvec)
   return function(input)
      if alphastd == 0 then
         return input
      end

      local alpha = torch.Tensor(3):normal(0, alphastd)
      local rgb = eigvec:clone()
         :cmul(alpha:view(1, 3):expand(3, 3))
         :cmul(eigval:view(1, 3):expand(3, 3))
         :sum(2)
         :squeeze()

      input = input:clone()
      for i=1,3 do
         input[i]:add(rgb[i])
      end
      return input
   end
end

local function blend(img1, img2, alpha)
   return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
   dst:resizeAs(img)
   dst[1]:zero()
   dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
   dst[2]:copy(dst[1])
   dst[3]:copy(dst[1])
   return dst
end

function M.Saturation(var)
   local gs

   return function(input)
      gs = gs or input.new()
      grayscale(gs, input)

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function M.Brightness(var)
   local gs

   return function(input)
      gs = gs or input.new()
      gs:resizeAs(input):zero()

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end


function M.AdditiveGausNoise(var_1, var_2)

    return function(input)
        local gs = input.new()
        gs:resizeAs(input):zero()
        -- print(gs:size())

        local sigma = torch.uniform(var_1, var_2)
        torch.normal(gs:select(1,1), 0, sigma)
            gs:select(1,2):copy(gs:select(1,1))
            gs:select(1,3):copy(gs:select(1,1))
            gs:select(1,4):copy(gs:select(1,1))
            gs:select(1,5):copy(gs:select(1,1))
            gs:select(1,6):copy(gs:select(1,1))
       
        return input:add(gs)
    end
end

function M.Contrast(var_1, var_2)

   return function(input)
      local gs = input.new()
      gs:resizeAs(input):zero()
    --   local ref_gray = rgb2gray(input[{ {1,3},{},{} }])
    --   local tar_gray = rgb2gray(input[{ {4,6},{},{} }])
      grayscale(gs[{ {1,3},{},{} }], input[{ {1,3},{},{} }])
      grayscale(gs[{ {4,6},{},{} }], input[{ {4,6},{},{} }])
      gs[{ {1,3},{},{} }]:fill(gs[{ {1,3},{},{} }][1]:mean())
      gs[{ {4,6},{},{} }]:fill(gs[{ {4,6},{},{} }][1]:mean())

      local alpha = 1.0 + torch.uniform(var_1, var_2)
      blend(input, gs, alpha)
      return input
   end
end

function M.MultiplicativeColorChange(var_1, var_2)

    return function(input)

      local mult_R = torch.uniform(var_1, var_2)
      local mult_G = torch.uniform(var_1, var_2)
      local mult_B = torch.uniform(var_1, var_2)

      input:select(1,1):mul(mult_R)
      input:select(1,2):mul(mult_G)
      input:select(1,3):mul(mult_B)
      input:select(1,4):mul(mult_R)
      input:select(1,5):mul(mult_G)
      input:select(1,6):mul(mult_B)


      return input
    end
end

function M.AdditiveBrightness(var)

    return function(input) 

      local ref_hsl = image.rgb2hsl(input[{ {1,3},{},{} }])
      local tar_hsl = image.rgb2hsl(input[{ {4,6},{},{} }])
      local changes = torch.normal(0, 0.2)
      ref_hsl:select(1,3):add(changes)
      tar_hsl:select(1,3):add(changes)
      input[{ {1,3},{},{} }]:copy(image.hsl2rgb(ref_hsl))
      input[{ {4,6},{},{} }]:copy(image.hsl2rgb(tar_hsl))

      return input
    end
end

function M.GammaChanges(var_1, var_2)

    return function(input) 
      local ref_hsl = image.rgb2hsl(input[{ {1,3},{},{} }])
      local tar_hsl = image.rgb2hsl(input[{ {4,6},{},{} }])
      local gamma   = torch.uniform(var_1, var_2)
      ref_hsl:select(1,3):pow(gamma)
      tar_hsl:select(1,3):pow(gamma)
      input[{ {1,3},{},{} }]:copy(image.hsl2rgb(ref_hsl))
      input[{ {4,6},{},{} }]:copy(image.hsl2rgb(tar_hsl))

      return input
    end
end


function M.Translations_naive(var)  
    return function(input)
       local inputSize = input:size()
       local trans_x = torch.random(-var*inputSize[3], var*inputSize[3])
       local trans_y = torch.random(-var*inputSize[3], var*inputSize[3])

      --  local x_from, x_to, y_from, y_to = 0,0,0,0


      --  if (trans_x <= 0) and (trans_y <= 0) then
      --     x_from  = 1
      --     x_to    = inputSize[3] + trans_x
      --     y_from  = 1
      --     y_to    = inputSize[2] + trans_y
      --  elseif (trans_x <= 0) and (trans_y > 0) then
      --     x_from  = 1
      --     x_to    = inputSize[3] + trans_x
      --     y_from  = trans_y
      --     y_to    = inputSize[2]
      --  elseif (trans_x > 0) and (trans_y <= 0) then
      --     x_from  = trans_x
      --     x_to    = inputSize[3]
      --     y_from  = 1
      --     y_to    = inputSize[2] + trans_y
      --  elseif (trans_x > 0) and (trans_y > 0) then
      --     x_from  = trans_x
      --     x_to    = inputSize[3]
      --     y_from  = trans_y
      --     y_to    = inputSize[2]
      --  end


      --  print('translations')
      -- --  print(image.crop(input, x_from, y_from, x_to, y_to):size())

      -- --  return image.crop(input, x_from, y_from, x_to, y_to)
      -- local translated_out = torch.zeros(input:size())
      --  return translated_out[{ {},{y_from,y_to},{x_from,x_to} }]
        return image.translate(input, trans_x, trans_y)
    end
end

function M.Translations_wo_blacks_relative(var)  
    return function(input)
       local inputSize = input:size()
       local trans_x = torch.random(-var*inputSize[3], var*inputSize[3])
       local trans_y = torch.random(-var*inputSize[3], var*inputSize[3])

       local translated_img = image.translate(input, trans_x, trans_y)
       return image.crop(translated_img, inputSize[3]/2 - 256, inputSize[2]/2 - 192, inputSize[3]/2 + 256, inputSize[2]/2 + 192)
    end
end

function M.Translations_wo_blacks(var)  
    return function(input)
       local inputSize = input:size()
       local trans_x = torch.random(-var*inputSize[3], var*inputSize[3])
       local trans_y = torch.random(-var*inputSize[3], var*inputSize[3])

      return image.translate(input, trans_x, trans_y)
    end
end

function M.HorizontalFlip(prob)
   return function(input)
      if torch.uniform() < prob then
       input = image.hflip(input)
      end
      return input
   end
end


function M.Rotation_naive(var)
   return function(input)
     local deg      = torch.uniform(-var, var)
 
     return image.rotate(input, deg * math.pi / 180, 'bilinear')
   end
end

function M.Rotation_wo_blacks_init(var)
   return function(input)
     local deg      = torch.uniform(-var, var)
 
     return image.rotate(input, deg * math.pi / 180, 'bilinear')
   end
end

function M.Rotation_wo_blacks_relative(var)
   return function(input)
     local inputSize = input:size()
     local deg       = torch.uniform(-var, var)

     local rotated_img = image.rotate(input, deg * math.pi / 180, 'bilinear')

     return image.crop(rotated_img, inputSize[3]/2 - 256, inputSize[2]/2 - 192, inputSize[3]/2 + 256, inputSize[2]/2 + 192)
   end
end

function M.Scales_naive(minSize, maxSize)
   return function(input)
      local w, h        = input:size(3), input:size(2)
      local factors     = torch.uniform(minSize, maxSize)
      local w1          = math.ceil(w*factors)
      local h1          = math.ceil(h*factors)
      local scaled      = image.scale(input, w1, h1)
      local scaled_input= torch.zeros(input:size(1),h,w)

      if factors > 1 then

        local center_x      = math.ceil(w1/2)
        local center_y      = math.ceil(h1/2)
        scaled_input        = image.crop(scaled, center_x - w/2, center_y - h/2, center_x + w/2, center_y + h/2)
     
      elseif factors < 1 then
  
        scaled_input[{ {},{1,h1},{1,w1} }] = scaled:clone()
  
      else
         
         scaled_input        = input:clone()

      end
      return scaled_input 
   end
end

function M.Scales_wo_blacks_relative(minSize, maxSize)
   return function(input)
      local w, h        = input:size(3), input:size(2)
      local factors     = torch.uniform(minSize, maxSize)
      local w1          = math.ceil(w*factors)
      local h1          = math.ceil(h*factors)
      local scaled      = image.scale(input, w1, h1)

      return image.crop(scaled, w1/2 - 256, h1/2 - 192, w1/2 + 256, h1/2 + 192)
   end
end


function M.Scales_wo_blacks(minSize, maxSize)
   return function(input)
      local w, h        = input:size(3), input:size(2)
      local factors     = torch.uniform(minSize, maxSize)
      local w1          = math.ceil(w*factors)
      local h1          = math.ceil(h*factors)
      
      return image.scale(input, w1, h1)
   end
end


function M.nCrop()
    return function(input)
       local inputSz     = input:size()
       if (inputSz[2] ~= inputSz[3]) then
          local imgSz       = math.min(inputSz[2] - 128,inputSz[3] - 128) --256
          local largeGaps   = math.floor(imgSz*0.4)                       --25
          local largeGaps_half = math.floor(largeGaps/2)                  --12
          local largeInputs = torch.zeros(3, imgSz + largeGaps, imgSz + largeGaps - 1) -- 3x280x280
          local rndPos_x    = torch.random(1 + largeGaps_half, inputSz[3] - largeGaps_half - imgSz) -- 13~244
          local rndPos_y    = torch.random(1 + largeGaps_half, inputSz[2] - largeGaps_half - imgSz) -- 13~116

          -- print ( rndPos_x - largeGaps_half,  rndPos_y - largeGaps_half,rndPos_x + (imgSz + largeGaps_half), rndPos_y + (imgSz + largeGaps_half))
          largeInputs       = image.crop(input, rndPos_x - largeGaps_half, rndPos_y - largeGaps_half, rndPos_x + (imgSz + largeGaps_half), rndPos_y + (imgSz + largeGaps_half))
          
          input             = image.crop(input, rndPos_x, rndPos_y, rndPos_x + imgSz, rndPos_y + imgSz)
          return input, largeInputs
       else
          return input
       end
    end
end

function M.centerCrop()
    return function(input)
       local inputSize     = input:size()
	   	 
       return image.crop(input, inputSize[3]/2 - 256, inputSize[2]/2 - 192, inputSize[3]/2 + 256, inputSize[2]/2 + 192)
    end
end

function M.randomCrop()
    return function(input)

       local inputSz     = input:size()
		local x_from      = torch.random(1,inputSz[3]-384)
       local y_from      = torch.random(1,inputSz[2]-256)

       return image.crop(input, x_from, y_from, x_from + 384, y_from + 256)
    end
end


function M.randomCrop_b()
    return function(b)
       local inputSz     = b:size()
       
       local x_from      = torch.random(1,inputSz[3]-384)
       local y_from      = torch.random(1,inputSz[2]-256)

       return image.crop(b, x_from, y_from, x_from + 384, y_from + 256)
    end
end

function M.randomCrop_a()
    return function(a)
       local inputSz     = a:size()
       
       local x_from      = torch.random(1,inputSz[3]-384)
       local y_from      = torch.random(1,inputSz[2]-256)

       return image.crop(a, x_from, y_from, x_from + 384, y_from + 256)
    end
end


function M.Identity()
   return function(input)
      return input
   end
end



return M
