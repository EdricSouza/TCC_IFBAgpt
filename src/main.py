import openai
import os
import dotenv

dotenv.load_dotenv()
token = os.getenv('openaisecret')

openai.api_key = token

#Gerar a resposta do chatgpt.
def Generate_Response(conversa):
    if not isinstance(conversa, list):
        conversa = [{"role":"user","content":conversa}]
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages=conversa,
        max_tokens=2056,
        temperature=1,
    )

    response = response.choices[0].message.content

    return response

#Continua uma conversa existente com o chatgpt
def conversation(message: str, conversation=[{"role":"system","content":"Você é uma boa IA."}]):
    conversation.append({"role":"user","content":message})
    response = Generate_Response(conversation)
    conversation.append({"role":"assistant","content":response})
    Return = [conversation,response]
    return Return

def Context(context: str):
    list = [{"role":"system","content":"Você é uma boa IA."},{"role":"user","content":"Analise todo o texto que vou lhe mandar a partir de agora."}]
    if len(context) >= 2056:
        voltas = 2056 // len(context)
        contador_voltas = 0
        for c in range(0,voltas):
            if contador_voltas < (voltas-1):
                fragmento = context[:2057]
                del context[:2057]

            else:
                fragmento = context

            list.append({"role":"assistant","content":"esperando texto do cliente."})
            list.append({"role":"user","content":f"continuação do texto: {fragmento}"})
            contador_voltas += 1
        return list
    
    else:
        list.append({"role":"assistant","content":"esperando texto do cliente."})
        list.append({"role":"user","content":fragmento})
        return list