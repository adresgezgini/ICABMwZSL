<div align="center">

# BERT Modeli'nin Sınıflandırma Doğruluğunun Sıfır-Atış Öğrenmesi ile Artırılması

</div>


  Türkiye Bilişim Vakfı Bilgisayar Bilimleri ve Mühendisliği Dergisi 1004781 no'lu BERT Modeli'nin Sınıflandırma Doğruluğunun Sıfır-Atış Öğrenmesi ile Artırılması isimli makalede kullanılan kaynakların genişletilmiş hali ve hata dizeyleri aşağıdaki gibidir. Bahsedilen analizler sırasında kullanılan kodlar açık kaynaklı olarak Python Codes klasörüne eklenmiştir. 


<div align="center">

Çizelge-4: Eldeki Veri Seti ile Doğrudan Eğitilen BERT Modeli ile Elde Edilen Kategorilere Göre Sınıflandırma Doğruluk Değerleri Çizelgesi

| ID    | Kesinlik   | Duyarlılık | F1-Skoru   | Doğruluk   |  Metin     |
|  :---: |  :---: |  :---: |  :---: |  :---: |  :---: |
|  41	|   0,94     |   0,93     |   0,94     |   0,93	    |   363      |
|  35	|   0,89     |   0,92     |   0,91     |   0,92	    |   424      |
|  31	|   0,85     |   0,92     |   0,88     |   0,92	    |   474      |
|  28	|   0,88     |   0,91     |   0,89     |   0,91	    |   507      |
|  27	|   0,85     |   0,90     |   0,87     |   0,90	    |   694      |
|  39	|   0,81     |   0,89     |   0,84     |   0,89	    |   840      |
|  12	|   0,86     |   0,88     |   0,87     |   0,88	    |   519      |
|  4 	|   0,92     |   0,88     |   0,90     |   0,88	    |   353      |
|  2 	|   0,89     |   0,86     |   0,88     |   0,86	    |   288      |
|  40	|   0,83     |   0,86     |   0,84     |   0,86	    |   560      |
|  25	|   0,90     |   0,85     |   0,87     |   0,85	    |   378      |
|  24	|   0,79     |   0,85     |   0,82     |   0,85	    |   414      |
|  37	|   0,80     |   0,84     |   0,82     |   0,84	    |   1036     |
|  33	|   0,89     |   0,83     |   0,86     |   0,83	    |   423      |
|  8 	|   0,83     |   0,82     |   0,82     |   0,82	    |   629      |
|  20	|   0,70     |   0,82     |   0,76     |   0,82	    |   1213     |
|  5 	|   0,83     |   0,82     |   0,82     |   0,82	    |   482      |
|  22	|   0,88     |   0,82     |   0,85     |   0,82	    |   432      |
|  19	|   0,80     |   0,81     |   0,80     |   0,81	    |   397      |
|  42	|   0,80     |   0,80     |   0,80     |   0,80	    |   225      |
|  18	|   0,88     |   0,80     |   0,84     |   0,80	    |   529      |
|  43	|   0,74     |   0,77     |   0,75     |   0,77	    |   471      |
|  17	|   0,76     |   0,77     |   0,76     |   0,77	    |   406      |
|  1 	|   0,85     |   0,76     |   0,80     |   0,76	    |   334      |
|  36	|   0,66     |   0,76     |   0,71     |   0,76	    |   1107     |
|  13	|   0,82     |   0,75     |   0,79     |   0,75	    |   431      |
|  30	|   0,80     |   0,74     |   0,77     |   0,74	    |   203      |
|  21	|   0,69     |   0,73     |   0,71     |   0,73	    |   235      |
|  10	|   0,83     |   0,72     |   0,77     |   0,72	    |   337      |
|  7 	|   0,79     |   0,72     |   0,76     |   0,72	    |   296      |
|  6 	|   0,78     |   0,71     |   0,75     |   0,71	    |   227      |
|  32	|   0,72     |   0,71     |   0,71     |   0,71	    |   282      |
|  14	|   0,78     |   0,70     |   0,74     |   0,70	    |   600      |
|  26	|   0,81     |   0,70     |   0,75     |   0,70	    |   233      |
|  15	|   0,72     |   0,70     |   0,71     |   0,70	    |   816      |
|  11	|   0,67     |   0,69     |   0,68     |   0,69	    |   252      |
|  3	|   0,78     |   0,69     |   0,73     |   0,69	    |   342      |
|  34	|   0,80     |   0,68     |   0,74     |   0,68	    |   282      |
|  29	|   0,79     |   0,68     |   0,73     |   0,68	    |   442      |
|  9 	|   0,67     |   0,68     |   0,67     |   0,68	    |   385      |
|  0 	|   0,76     |   0,67     |   0,71     |   0,67	    |   236      |
|  23	|   0,71     |   0,67     |   0,69     |   0,67	    |   338      |
|  16	|   0,73     |   0,65     |   0,69     |   0,65	    |   462      |
|  38	|   0,79     |   0,62     |   0,70     |   0,62	    |   306      |
|     |            |            |            |            |            |
|   	|   0,79     |   0,79     |   0,79     |   0,79     |  20203     |
|   	| (Ağ. Ort)  | (Ağ. Ort)  | (Ağ. Ort)  | (Ağ. Ort)  | (Toplam)   |
</div>


&nbsp;

  
  
<div align="center">
  
Eldeki Veri Seti ile Doğrudan Eğitilen BERT Modeli ile Elde Edilen Kategorilere Göre Sınıfla Hata Dizeyi
  
!['Zero Shotsız BERT CONFİSİON'](https://github.com/adresgezgini/makale/blob/main/resource/Just_BERT_CONF.png)
</div>
&nbsp;
<div align="center">

Çizelge-5: Sıfır-Atış Öğrenmesi Yöntemi ile Aykırı Verilerden Arındırılmış Veri Seti İle Eğitilen BERT Modeli ile Elde Edilen Kategorilere göre Sınıflandırma Doğruluk Çizelgesi
  
| ID    | Kesinlik   | Duyarlılık | F1-Skoru   | Doğruluk   |  Metin     |
|  :---: |  :---: |  :---: |  :---: |  :---: |  :---: |
|  0	|   1,00     |   1,00     |   1,00     |   1,00     |   75       |
|  2	|   0,99     |   1,00     |   1,00     |   1,00     |   100      |
|  4	|   1,00     |   1,00     |   1,00     |   1,00     |   133      |
|  5	|   0,90     |   1,00     |   0,95     |   1,00     |   19       |
|  10	|   1,00     |   1,00     |   1,00     |   1,00     |   27       |
|  11	|   0,99     |   1,00     |   0,99     |   1,00     |   81       |
|  12	|   0,98     |   1,00     |   0,99     |   1,00     |   62       |
|  24	|   0,99     |   1,00     |   1,00     |   1,00     |   113      |
|  25	|   1,00     |   1,00     |   1,00     |   1,00     |   196      |
|  29	|   1,00     |   1,00     |   1,00     |   1,00     |   37       |
|  31	|   0,99     |   1,00     |   1,00     |   1,00     |   136      |
|  33	|   1,00     |   1,00     |   1,00     |   1,00     |   59       |
|  34	|   0,98     |   1,00     |   0,99     |   1,00     |   84       |
|  40	|   1,00     |   1,00     |   1,00     |   1,00     |   57       |
|  20	|   0,99     |   1,00     |   0,99     |   1,00     |   278      |
|  35	|   1,00     |   0,99     |   1,00     |   1,00     |   200      |
|  41	|   0,99     |   0,99     |   0,99     |   0,99     |   198      |
|  19	|   0,99     |   0,99     |   0,99     |   0,99     |   112      |
|  28	|   0,99     |   0,99     |   0,99     |   0,99     |   207      |
|  7	|   1,00     |   0,99     |   0,99     |   0,99     |   83       |
|  43	|   1,00     |   0,99     |   0,99     |   0,99     |   70       |
|  8	|   0,99     |   0,98     |   0,99     |   0,98     |   128      |
|  17	|   1,00     |   0,98     |   0,99     |   0,98     |   61       |
|  39	|   0,93     |   0,98     |   0,96     |   0,98     |   56       |
|  37	|   1,00     |   0,98     |   0,99     |   0,98     |   218      |
|  1	|   1,00     |   0,98     |   0,99     |   0,98     |   96       |
|  22	|   0,98     |   0,98     |   0,98     |   0,98     |   84       |
|  23	|   0,94     |   0,98     |   0,96     |   0,98     |   82       |
|  15	|   0,97     |   0,97     |   0,97     |   0,98     |   160      |
|  30	|   0,91     |   0,97     |   0,94     |   0,98     |   40       |
|  18	|   0,96     |   0,97     |   0,97     |   0,97     |   78       |
|  3	|   0,95     |   0,97     |   0,96     |   0,97     |   38       |
|  14	|   0,91     |   0,97     |   0,94     |   0,97     |   32       |
|  26	|   0,97     |   0,97     |   0,97     |   0,97     |   64       |
|  32	|   1,00     |   0,97     |   0,98     |   0,97     |   30       |
|  16	|   0,92     |   0,96     |   0,94     |   0,96     |   49       |
|  36	|   0,91     |   0,96     |   0,93     |   0,96     |   91       |
|  13	|   1,00     |   0,95     |   0,97     |   0,95     |   61       |
|  21	|   0,95     |   0,95     |   0,95     |   0,95     |   20       |
|  27	|   1,00     |   0,95     |   0,97     |   0,95     |   20       |
|  42	|   0,97     |   0,93     |   0,95     |   0,93     |   80       |
|  6	|   0,88     |   0,88     |   0,88     |   0,88     |   17       |
|  38	|   0,96     |   0,82     |   0,89     |   0,82     |   33       |
|  9	|   0,88     |   0,69     |   0,77     |   0,69     |   32       |
|     |            |            |            |            |            |
|   	|   0,98     |   0,98     |   0,98     |   0,98     |  3897      |
|   	| (Ağ. Ort)  | (Ağ. Ort)  | (Ağ. Ort)  | (Ağ. Ort)  | (Toplam)   |
</div>
&nbsp;
<div align="center">
Sıfır-Atış Öğrenmesi Yöntemi ile Aykırı Verilerden Arındırılmış Veri Seti İle Eğitilen BERT Modeli ile Elde Edilen Kategorilere göre Sınıflandırma Hata Dizeyi

&nbsp;

!['Zero Shotlu BERT CONFİSİON'](https://github.com/adresgezgini/makale/blob/main/resource/Zero%2BBERT_CONF.png)

</div>

