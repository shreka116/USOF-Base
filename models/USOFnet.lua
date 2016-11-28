--
--  Copyright (c) 2016, Computer Vision Lab @ Seoul National University.
--  All rights reserved.
--
--  Architecture borrowed from FlowNet:Simple
--
--  Fischer, Philipp, et al. "Flownet: Learning optical flow with convolutional networks."
--  arXiv preprint arXiv:1504.06852 (2015).
--

--require 'nn'
require 'nngraph'
require 'cunn'
require 'cudnn'
require 'math'
local nninit = require 'nninit'

local numChannels = 3
local Conv      = cudnn.SpatialConvolution
local deConv    = cudnn.SpatialFullConvolution
local ReLU      = cudnn.ReLU
local MaxPool   = cudnn.SpatialMaxPooling
local UpSample  = nn.SpatialUpSamplingNearest

-- local f_depths  = {6,64,128,256,256,512,512,512,512,1024,512,256,128,64,2} -- feature depths
-- local f_depths  = {6,64,128,256,256,512,512,512,512,1024,512*2 + 2,512*2 + 2,256*2 +2,128*2 + 2,64*2 + 1,2} -- feature depths
-- local k_size    = { 7, 5,  5,  3,  3,  3,  3,  3,  3,   1,        1,        1,       1,        1,       1 } -- kernel size
local f_depths  = {2*numChannels,64,128,256,256,512,512,512,512,1024,512,512,256,128,64,2} -- feature depths
local k_size    = { 7, 5,  5,  3,  3,  3,  3,  3,  3,   1,  1,  1,  1,  1, 1 } -- kernel size

local function createModel(opt)
    
--     local input = nn.Identity()()
--     local Conv1 =  input
--                 - Conv(f_depths[1],f_depths[2],k_size[1],k_size[1],1,1,math.floor(k_size[1]/2),math.floor(k_size[1]/2))
--                 - MaxPool(2,2)
--                 - ReLU()
--     local Conv2 =  Conv1
--                 - Conv(f_depths[2],f_depths[3],k_size[2],k_size[2],1,1,math.floor(k_size[2]/2),math.floor(k_size[2]/2))
--                 - MaxPool(2,2)
--                 - ReLU()
--     local Conv3 =  Conv2
--                 - Conv(f_depths[3],f_depths[4],k_size[3],k_size[3],1,1,math.floor(k_size[3]/2),math.floor(k_size[3]/2))
--                 - MaxPool(2,2)
--                 - ReLU()
--     local Conv4 =  Conv3
--                 - Conv(f_depths[4],f_depths[5],k_size[4],k_size[4],1,1,math.floor(k_size[4]/2),math.floor(k_size[4]/2))
--                 - ReLU()
--     local Conv5 =  Conv4
--                 - Conv(f_depths[5],f_depths[6],k_size[5],k_size[5],1,1,math.floor(k_size[5]/2),math.floor(k_size[5]/2))
--                 - MaxPool(2,2)
--                 - ReLU()
--     local Conv6 =  Conv5
--                 - Conv(f_depths[6],f_depths[7],k_size[6],k_size[6],1,1,math.floor(k_size[6]/2),math.floor(k_size[6]/2))
--                 - ReLU()
--     local Conv7 =  Conv6
--                 - Conv(f_depths[7],f_depths[8],k_size[7],k_size[7],1,1,math.floor(k_size[7]/2),math.floor(k_size[7]/2))
--                 - MaxPool(2,2)
--                 - ReLU()
--     local Conv8 =  Conv7
--                 - Conv(f_depths[8],f_depths[9],k_size[8],k_size[8],1,1,math.floor(k_size[8]/2),math.floor(k_size[8]/2))
--                 - ReLU()
--     local Conv9 =  Conv8
--                 - Conv(f_depths[9],f_depths[10],k_size[9],k_size[9],1,1,math.floor(k_size[9]/2),math.floor(k_size[9]/2))
--                 - MaxPool(2,2)
--                 - ReLU()


--     local deConv0       = Conv9
--                         - deConv(f_depths[10],f_depths[11],k_size[10],k_size[10],2,2,0,0,1,1)
--                         - ReLU()
    
--     local joinedFeat1   = { Conv8 , deConv0 }
--                         - nn.JoinTable(2)
--     local sideOutput1   = joinedFeat1
--                         - UpSample(2)
--                         - Conv(f_depths[11]+f_depths[9],2,5,5,1,1,2,2)
--     local deConv1       = joinedFeat1
--                         - deConv(f_depths[11]+f_depths[9],f_depths[12],k_size[11],k_size[11],2,2,0,0,1,1)
--                         - ReLU()

--     local joinedFeat2   = { Conv6 , deConv1 , sideOutput1 }
--                         - nn.JoinTable(2)
--     local sideOutput2   = joinedFeat2
--                         - UpSample(2)
--                         - Conv(f_depths[12]+f_depths[7]+2,2,5,5,1,1,2,2)
--     local deConv2       = joinedFeat2
--                         - deConv(f_depths[12]+f_depths[7]+2,f_depths[13],k_size[12],k_size[12],2,2,0,0,1,1)
--                         - ReLU()

--     local joinedFeat3   = { Conv4 , deConv2 , sideOutput2 }
--                         - nn.JoinTable(2)
--     local sideOutput3   = joinedFeat3
--                         - UpSample(2)
--                         - Conv(f_depths[13]+f_depths[5]+2,2,5,5,1,1,2,2)
--     local deConv3       = joinedFeat3
--                         - deConv(f_depths[13]+f_depths[5]+2,f_depths[14],k_size[13],k_size[13],2,2,0,0,1,1)
--                         - ReLU()

--     local joinedFeat4   = { Conv2 , deConv3 , sideOutput3 }
--                         - nn.JoinTable(2)
--     local sideOutput4   = joinedFeat4
--                         - UpSample(2)
--                         - Conv(f_depths[14]+f_depths[3]+2,2,5,5,1,1,2,2)
--     local deConv4       = joinedFeat4
--                         - deConv(f_depths[14]+f_depths[3]+2,f_depths[15],k_size[14],k_size[14],2,2,0,0,1,1)
--                         - ReLU()                        

--     local joinedFeat5   = { Conv1 , deConv4 , sideOutput4 }
--                         - nn.JoinTable(2)
--     -- sideOutput5 is the actual output of the network                        
--     local sideOutput5   = joinedFeat5
--                         - UpSample(2)
--                         - Conv(f_depths[15]+f_depths[2]+2,f_depths[16],5,5,1,1,2,2)


--    local model = nn.gModule({input} , {sideOutput5})     

-- define conv9 followed by deconv0
local conv9_deconv0 = nn.Sequential()
conv9_deconv0:add(Conv(f_depths[9],f_depths[10],k_size[9],k_size[9],2,2,math.floor(k_size[9]/2),math.floor(k_size[9]/2)))
--conv9_deconv0:add(MaxPool(2,2))
conv9_deconv0:add(ReLU(true))
conv9_deconv0:add(deConv(f_depths[10],f_depths[11],k_size[10],k_size[10],2,2,0,0,1,1))
conv9_deconv0:add(ReLU(true))

-- define concatTable between conv8 and conv9_deconv0
local concat_8_9 = nn.ConcatTable()
concat_8_9:add(nn.Identity())
concat_8_9:add(conv9_deconv0)

-- define conv7 followed by conv8
local conv7_8 = nn.Sequential()
conv7_8:add(Conv(f_depths[7],f_depths[8],k_size[7],k_size[7],2,2,math.floor(k_size[7]/2),math.floor(k_size[7]/2)))
--conv7_8:add(MaxPool(2,2))
conv7_8:add(ReLU(true))
conv7_8:add(Conv(f_depths[8],f_depths[9],k_size[8],k_size[8],1,1,math.floor(k_size[8]/2),math.floor(k_size[8]/2)))
conv7_8:add(ReLU(true))
conv7_8:add(concat_8_9)
conv7_8:add(nn.JoinTable(2))
-- define concat of deconv1 and sideoutput1
local seq_deconv1_side1     = nn.Sequential()
local concat_deconv1_side1  = nn.ConcatTable()
local sideoutput_1          = nn.Sequential()
local deconv_1              = nn.Sequential()
deconv_1:add(deConv(f_depths[11]+f_depths[9],f_depths[12],k_size[11],k_size[11],2,2,0,0,1,1)) 
deconv_1:add(ReLU(true))
sideoutput_1:add(UpSample(2))
sideoutput_1:add(Conv(f_depths[11]+f_depths[9],2,5,5,1,1,2,2))
concat_deconv1_side1:add(deconv_1)
concat_deconv1_side1:add(sideoutput_1)
seq_deconv1_side1:add(concat_deconv1_side1)
seq_deconv1_side1:add(nn.JoinTable(2))

conv7_8:add(seq_deconv1_side1)

-- define concatTable between conv5_6 and conv7_8
local concat_6_7 = nn.ConcatTable()
concat_6_7:add(nn.Identity())
concat_6_7:add(conv7_8)

local conv5_6 = nn.Sequential()
conv5_6:add(Conv(f_depths[5],f_depths[6],k_size[5],k_size[5],2,2,math.floor(k_size[5]/2),math.floor(k_size[5]/2)))
--conv5_6:add(MaxPool(2,2))
conv5_6:add(ReLU(true))
conv5_6:add(Conv(f_depths[6],f_depths[7],k_size[6],k_size[6],1,1,math.floor(k_size[6]/2),math.floor(k_size[6]/2)))
conv5_6:add(ReLU(true))
conv5_6:add(concat_6_7)
conv5_6:add(nn.JoinTable(2))
-- define concat of deconv2 and sideoutput2
local seq_deconv2_side2     = nn.Sequential()
local concat_deconv2_side2  = nn.ConcatTable()
local sideoutput_2          = nn.Sequential()
local deconv_2              = nn.Sequential()
deconv_2:add(deConv(f_depths[12]+f_depths[7]+2,f_depths[13],k_size[12],k_size[12],2,2,0,0,1,1))
deconv_2:add(ReLU(true))
sideoutput_2:add(UpSample(2))
sideoutput_2:add(Conv(f_depths[12]+f_depths[7]+2,2,5,5,1,1,2,2))
concat_deconv2_side2:add(deconv_2)
concat_deconv2_side2:add(sideoutput_2)
seq_deconv2_side2:add(concat_deconv2_side2)
seq_deconv2_side2:add(nn.JoinTable(2))

conv5_6:add(seq_deconv2_side2)

-- define concatTable between conv3_4 and conv5_6
local concat_4_5 = nn.ConcatTable()
concat_4_5:add(nn.Identity())
concat_4_5:add(conv5_6)

local conv3_4 = nn.Sequential()
conv3_4:add(Conv(f_depths[3],f_depths[4],k_size[3],k_size[3],2,2,math.floor(k_size[3]/2),math.floor(k_size[3]/2)))
--conv3_4:add(MaxPool(2,2))
conv3_4:add(ReLU(true))
conv3_4:add(Conv(f_depths[4],f_depths[5],k_size[4],k_size[4],1,1,math.floor(k_size[4]/2),math.floor(k_size[4]/2)))
conv3_4:add(ReLU(true))
conv3_4:add(concat_4_5)
conv3_4:add(nn.JoinTable(2))
-- define concat of deconv3 and sideoutput3
local seq_deconv3_side3     = nn.Sequential()
local concat_deconv3_side3  = nn.ConcatTable()
local sideoutput_3          = nn.Sequential()
local deconv_3              = nn.Sequential()
deconv_3:add(deConv(f_depths[13]+f_depths[5]+2,f_depths[14],k_size[13],k_size[13],2,2,0,0,1,1))
deconv_3:add(ReLU(true))
sideoutput_3:add(UpSample(2))
sideoutput_3:add(Conv(f_depths[13]+f_depths[5]+2,2,5,5,1,1,2,2))
concat_deconv3_side3:add(deconv_3)
concat_deconv3_side3:add(sideoutput_3)
seq_deconv3_side3:add(concat_deconv3_side3)
seq_deconv3_side3:add(nn.JoinTable(2))

conv3_4:add(seq_deconv3_side3)

-- define concatTable between conv2 and conv3_4
local concat_2_3 = nn.ConcatTable()
concat_2_3:add(nn.Identity())
concat_2_3:add(conv3_4)

local conv2 = nn.Sequential()
conv2:add(Conv(f_depths[2],f_depths[3],k_size[2],k_size[2],2,2,math.floor(k_size[2]/2),math.floor(k_size[2]/2)))
--conv2:add(MaxPool(2,2))
conv2:add(ReLU(true))
conv2:add(concat_2_3)
conv2:add(nn.JoinTable(2))
-- define concat of deconv4 and sideoutput4
local seq_deconv4_side4     = nn.Sequential()
local concat_deconv4_side4  = nn.ConcatTable()
local sideoutput_4          = nn.Sequential()
local deconv_4              = nn.Sequential()
deconv_4:add(deConv(f_depths[14]+f_depths[3]+2,f_depths[15],k_size[14],k_size[14],2,2,0,0,1,1))
deconv_4:add(ReLU(true))
sideoutput_4:add(UpSample(2))
sideoutput_4:add(Conv(f_depths[14]+f_depths[3]+2,2,5,5,1,1,2,2))
concat_deconv4_side4:add(deconv_4)
concat_deconv4_side4:add(sideoutput_4)
seq_deconv4_side4:add(concat_deconv4_side4)
seq_deconv4_side4:add(nn.JoinTable(2))

conv2:add(seq_deconv4_side4)

-- define concatTable between conv1 and conv2
local concat_1_2 = nn.ConcatTable()
concat_1_2:add(nn.Identity())
concat_1_2:add(conv2)

local model = nn.Sequential()
model:add(Conv(f_depths[1],f_depths[2],k_size[1],k_size[1],2,2,math.floor(k_size[1]/2),math.floor(k_size[1]/2)))
--model:add(MaxPool(2,2))
model:add(ReLU(true))
model:add(concat_1_2)
model:add(nn.JoinTable(2))
model:add(UpSample(2))
model:add(Conv(f_depths[15]+f_depths[2]+2,f_depths[16],5,5,1,1,2,2))


   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')



    return model
end

return createModel
