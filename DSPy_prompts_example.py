import os
import dspy

"""Configuring the LM"""
lm = dspy.LM('ollama_chat/llama3.2:1b-instruct-q6_K', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

"""Letting the user know the LM is configured."""
print("*************************************************" + os.linesep + os.linesep + "Configured the LM" + os.linesep + os.linesep + "*************************************************",)

"""Prompting the User for the LM's prompt."""
print("*************************************************" + os.linesep + os.linesep + "Obtaining user prompt." + os.linesep + os.linesep + "*************************************************",)

input_message = 'Talk to Llama: '
print("*************************************************" + os.linesep + os.linesep )

prompt = input(input_message)

print(os.linesep + os.linesep + "*************************************************")

"""Letting the user know its sending the prompt to the LM."""
print("*************************************************" + os.linesep + os.linesep + "Sending prompt to LM: " + os.linesep + prompt + os.linesep + os.linesep + "*************************************************",)

print("*************************************************" + os.linesep + os.linesep + "Calling the LM directly" + os.linesep + os.linesep + "*************************************************",)


# Calling the LM directly
output_1 = lm(prompt, temperature=0.7)  # => ['This is a test!']

output_2 = lm(messages=[{"role": "user", "content": prompt}])  # => ['This is a test!']

print("*************************************************" + os.linesep + os.linesep + "Printing the LM's Output" + os.linesep + os.linesep + "*************************************************",)

print(output_1[0])
print(output_2[0])

