import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from selenium.webdriver import Chrome
from collections import deque

def get_text_from_url(urls):
    '''
        Essa função recebe uma url e faz o scraping do texto da página
        Retorna o texto da página e salva o texto em um arquivo <url>.txt
    '''

    # Analisa a URL e pega o domínio
    local_domain = urlparse(url).netloc
    print(local_domain)

    # Fila para armazenar as urls para fazer o scraping
    fila = deque(url)
    print(fila)

    # Criar um diretório para armazenar os arquivos de texto
    if not os.path.exists("text/"):
            os.mkdir("text/")

    if not os.path.exists("text/"+local_domain+"/"):
            os.mkdir("text/" + local_domain + "/")

    # Create a directory to store the csv files
    if not os.path.exists("processed"):
            os.mkdir("processed")

    # Enquanto a fila não estiver vazia, continue fazendo o scraping
    while fila:
        # Pega a próxima URL da fila
        url = fila.pop()
        print("Próxima url",url) # Checa próxima url

        # Salva o texto da url em um arquivo <url>.txt
        with open('text/'+local_domain+'/'+url[8:].replace("/", "_") + ".txt", "w") as f:
            driver = Chrome()
            driver.get(url)
            page_soup = BeautifulSoup(driver.page_source, 'html.parser')
            text = page_soup.get_text()
            f.write(text)

        driver.close()





def remove_newlines(serie):
    '''
        Essa função recebe uma série e remove as quebras de linha
    '''
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie