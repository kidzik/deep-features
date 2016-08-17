require 'torch'
require 'nn'
require 'image'
require 'loadcaffe'

local cmd = torch.CmdLine()
cmd:option('-images_path', '/home/lukasz/Downloads/videoframes/', 'Path to images')
cmd:option('-output_csv', '/home/lukasz/Dropbox/output.csv', 'Path to output file')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-ext', 'jpg', 'Extension')

local function main(params)
   local cnn = loadcaffe.load(params.proto_file, params.model_file, 'nn')
   print('Loading models...\t[OK]')

   -- Create empty table to store file names:
   files = {}

   -- Go over all files in directory. We use an iterator, paths.files().
   for file in paths.files(params.images_path) do
      -- We only load files that match the extension
      if file:find(params.ext .. '$') then
	 -- and insert the ones we care about in our table
	 table.insert(files, paths.concat(params.images_path,file))
      end
   end

   print('Loading paths...\t[OK]')
   images = {}

   for i,file in ipairs(files) do
      -- load each image
      table.insert(images, image.load(file))
   end

   print('Loading images...\t[OK]')
   features = {}

   -- Go through the neural net
   for i = 1,#images do
      images[i] = image.scale(images[i],224,224)
      table.insert(features, cnn:forward(images[i]))
   end

   print('Extracting features...\t[OK]')

   -- Write a CSV
   local file = io.open(params.output_csv, "w")
   for i = 1,#images do
      file:write(files[i], ",")
      for j=1,features[i]:size(1) do
	 file:write(features[i][j])
	 if j == features[i]:size(1) then
            file:write("\n")
	 else
            file:write(",")
	 end
      end
   end
   file:close()

   print('Writing CSV...\t\t[OK]')
end

local params = cmd:parse(arg)
main(params)

