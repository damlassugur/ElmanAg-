import numpy as np
import matplotlib.pyplot as plt

def sigma(x):
    return np.tanh(x)


def tursigma(x):
    return 1.0-x**2


def billings(a,b):
    y=()
    y=np.array(y)
    y=np.append(y,a)
    y=np.append(y,b)
    y=np.append(y,(0.8-0.5*np.exp(-y[1]**2))*y[1]-(0.3+0.9*np.exp(-y[1]**2)*y[0]+0.1*np.sin(np.pi*y[1])))
    return y[-1]


class Elman:
    "Elman Ağının Algoritması"

    def __init__(self, *args):
        "Başlangıç değişkenleri"

        self.shape = args
        n = len(args)

        #layerların oluşumu
        self.katman = []

        
        self.katman.append(np.ones(self.shape[0]+1+self.shape[1]))

        #gizli katman ve çıkış katmanı
        for i in range(1,n):
            self.katman.append(np.ones(self.shape[i]))

        #Ağırlık vektörlerinin oluşumu
        self.w= []
        for i in range(n-1):
            self.w.append(np.zeros((self.katman[i].size,
                                    self.katman[i+1].size)))

        
        self.mom = [0,]*len(self.w)

        # Reset
        self.reset()


    def reset(self):
        ''' Ağırlıkların random atanması '''

        for i in range(len(self.w)):
            Z = np.random.random((self.katman[i].size,self.katman[i+1].size))
            self.w[i][...] = (2*Z-1)*0.25


    def ileriyol(self, data):
        '''İleri yol fonksiyonu '''

        # Datanın girişe verilmesi
        self.katman[0][:self.shape[0]] = data
        # Gizli katman
        self.katman[0][self.shape[0]:-1] = self.katman[1]

      
        for i in range(1,len(self.shape)):
          
            self.katman[i][...] = sigma(np.dot(self.katman[i-1],self.w[i-1]))

        return self.katman[-1]


    def geriyol(self, target, lrate=0.5, momentum=0.1):
        ''' Geri Besleme fonksiyonu, Learning rate ile'''

        deltas = []

        # Hata hesabı
        error = target - self.katman[-1]
        delta = error*tursigma(self.katman[-1])
        deltas.append(delta)

        # gizli katman hata hesabı
        for i in range(len(self.shape)-2,0,-1):
            delta = np.dot(deltas[0],self.w[i].T)*tursigma(self.katman[i])
            deltas.insert(0,delta)
            
        # Ağırlık güncellenmesi
        for i in range(len(self.w)):
            katmank = np.atleast_2d(self.katman[i])
            delta = np.atleast_2d(deltas[i])
            mom = np.dot(katmank.T,delta)
            self.w[i] += lrate*mom + momentum*self.mom[i]
            self.mom[i] = mom
       
        return (error**2).sum(),(error**2)/2


#Billings sistemi için dataların oluşturulması
data = np.zeros(100, dtype=[('input',  float, 2), ('output', float, 1)])
y=()
y=np.array(y)
data[0]=(3,3) , billings(3,3) #başlangıç için iki geçmiş değer girilmesi ve çıkışın billings fonksiyonu ile hesaplanması
y=np.append(y, data['input'][0])
y=np.append(y, data['output'][0])


for i in range(1,100): #100 terimin gerçek çıkış değerlerinin fonksiyon ile hesaplanması
    data[i]=(y[i],y[i+1]), billings(y[i],y[i+1])
    y=np.append(y,data['output'][i])
#Gerçek fonksiyon değerlerinin çizilmesi
plt.figure()
plt.plot(sigma(y))
plt.legend('data')
#classtan ağın yaratılması
network = Elman(2,18,1)
#öğrenme aşaması

for i in range(1000):
    n = i%data.size
    network.ileriyol(sigma(data['input'][n])) #ileri yol fonkisyonu
    (a,error)=network.geriyol(sigma(data['output'][n])) #geri yol fonksiyonu

#öğrenmenin ardından ağın sonuçlarının tutulması amacıyla array oluşturulması
ys=()
ys=np.array(ys)
ys=np.append(ys,sigma(data['input'][0]))
ys=np.append(ys,sigma(data['output'][0]))
#ağın tahminleri
for i in range(data.size):
    o = network.ileriyol(sigma(data['input'][i] )) #sadece ileri yol fonk çalıştırılıyor
    print ('Data %d: %s -> %s' % (i, sigma(data['input'][i]), sigma(data['output'][i])))
    print ( "                         Ağın tahmini:",o)
    ys=np.append(ys,o)

error=((data['output']-ys[3:103])**2)/2
#Ağın tahminlerinin çizilmesi
plt.plot(ys)
plt.legend('tahmin')
plt.title("Ağın tahminleri ve gerçek değerler")
plt.xlabel("k")
plt.ylabel("y(k) Değerleri")
plt.show()
plt.figure()
plt.plot(error)
plt.title("Hata")
plt.show()
plt.figure()

