from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model # the main line which will allow any LLM model to be used
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

model = init_chat_model("mistral-small-latest",
                        temperature=0
                        )

print("Choose your AI mode")
print("Press 1 for Angry mode")
print("Press 2 for Funny mode")
print("Press 3 for Sad mode")

choice = int(input("Tell your response:-"))

if choice==1:
    mode = "You are an angry AI agent. You respond aggressively and impatiently."

elif choice==2:
    mode="You are a very funny AI agent. You respond with humor and jokes"

elif choice==3:
    mode="You are a very sad AI agent. You respond with sadness"

messages=[
    SystemMessage(content=mode)
]

print("------------Welcome type 0 to exit the application--------------")

while True:
    
    prompt = input("You: ")
    messages.append(HumanMessage(content=prompt))
    if prompt=='0':
        break

    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Bot: ",response.content)