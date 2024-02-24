import urllib.request as urlreq
    
def countWordsOfUrl(url):
    try:
        response = urlreq.urlopen(url)
        
        text = response.read().decode('utf-8')
        
        words = text.split()
        print(words)
        return len(words)
    
    except Exception as e:
        print("Error:", e)
        return None
   

def main():
    url = "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"
    #2 menos
    numWord = countWordsOfUrl(url)
    
    if numWord is not None:
        print("El número de palabras en la página es:", numWord)
    else:
        print("No se pudo obtener el número de palabras")

if __name__ == "__main__":
    main()