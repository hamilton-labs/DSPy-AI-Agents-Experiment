import os
import dspy
lm = dspy.LM('ollama_chat/llama3.2:1b-instruct-q6_K', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)
print("*************************************************" + os.linesep + os.linesep + "Configured the LM" + os.linesep + os.linesep + "*************************************************",)

print("*************************************************" + os.linesep + os.linesep + "Calling the LM directly" + os.linesep + os.linesep + "*************************************************",)
# Calling the LM directly
output_1 = lm("Say this is a test!", temperature=0.7)  # => ['This is a test!']

output_2 = lm(messages=[{"role": "user", "content": "Say this is a test!"}])  # => ['This is a test!']

print("*************************************************" + os.linesep + os.linesep + "Printing the Lm's Output" + os.linesep + os.linesep + "*************************************************",)

print(output_1[0])
print(output_2[0])
