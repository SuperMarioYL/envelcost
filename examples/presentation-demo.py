import json
from envelcost.envelope import ToolDef
from envelcost.tokenizer import Tokenizer
tool=ToolDef(name='read',description='Read a file',parameters={'type':'object','properties':{'path':{'type':'string'}},'required':['path']})
t=Tokenizer()
for name,text in [('compact',tool.native_block()),('json-schema',tool.openai_block())]:
 print(json.dumps({'representation':name,'text':text,'tokens_same_cl100k':t.count(text,'openai')},ensure_ascii=False))
print('cl100k available:',t.openai_available)
