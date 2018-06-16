-- We thank Kai Sheng Tai for providing this preprocessing codes

require('torch')
require('xlua')

--local path = arg[1]
local vocabpath = arg[1]
local vecpath = arg[2]
--local prefix_toks = stringx.split(path, '.')
print('Converting (piped data) to Torch serialized format (storing in ' .. vecpath .. ')')

-- get dimension and number of lines
--local file = io.open(path, 'r')
local line
local count = 0
local dim = 0
--while true do
--  line = file:read()
--  if not line then break end
--  if count == 0 then
--    dim = #stringx.split(line) - 1
--  end
--  count = count + 1
--end

local firstline = io.read()
count, dim = string.match(firstline, "([0-9]+) ([0-9]+)")
print('count = ' .. count)
print('dim = ' .. dim)
count = tonumber(count)
dim = tonumber(dim)

-- convert to torch-friendly format
--file:seek('set')
local vocab = io.open(vocabpath, 'w')
local vecs = torch.FloatTensor(count, dim)
local i = 1
--for i = 1, count do
for line in io.lines() do
  xlua.progress(i, count)
  local tokens = stringx.split(line)
  local word = tokens[1]
  vocab:write(word .. '\n')
  for j = 1, dim do
    vecs[{i, j}] = tonumber(tokens[j + 1])
  end
  i = i+1
end
--file:close()
vocab:close()
torch.save(vecpath, vecs)
