# 가수 한 명을 탄생

# import pack2.class4
from pack2.class4 import SingerType

def process():
    # youngwoung = pack2.class4.SingerType()
    youngwoung = SingerType()
    print('영웅의 타이틀 송 : ', youngwoung.title_song)
    youngwoung.sing()
    
def process2():
    bts = SingerType()
    bts.sing()
    bts.title_song = '최고의 순간은 아직'
    bts.sing()
    bts.co = 'HIVE'
    print('소속사 : ', bts.co)
    blackpink = SingerType()
    blackpink.sing()
    blackpink.title_song = '셧다운'
    blackpink.sing()
    # print('소속사 : ', blackpink.co) #'SingerType' object has no attribute 'co'
    
if __name__ == '__main__':
    process()
    print('-----------')
    process2()

    