import pandas as pd
import matplotlib.pyplot as plt


def main():
    ruta = "./Lab4/exercise5/company_sales_data.csv"
    compDF = pd.read_csv(ruta)

    months = compDF['month_number']
    faceCream = compDF['facecream']
    faceWash = compDF['facewash']
    
    plt.figure(figsize=(10, 6))
    plt.bar(months, faceCream, width=-0.4, align='edge', label='Face Cream sales data' )
    plt.bar(months, faceWash, width=0.4, align='edge', label='Face wash sales data')
    plt.xlabel('Month Number')
    plt.ylabel('Sales units in number')
    plt.title('Facewash and facecream sales data')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()



    


if __name__ == "__main__":
    main()
