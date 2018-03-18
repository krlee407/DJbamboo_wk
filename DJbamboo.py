import re
from konlpy.tag import Twitter
import math
import heapq, random
import csv
import numpy as np
import os

def comeondata():
    with open('./data/fam.csv','r') as f:
        reader = csv.reader(f)
        global fam
        fam = list(reader)
        fam = np.array(fam)
        fam = np.float64(fam)

    with open('./data/lov.csv','r') as f:
        reader = csv.reader(f)
        global lov
        lov = list(reader)
        lov = np.array(lov)
        lov = np.float64(lov)

    with open('./data/sdf.csv','r') as f:
        reader = csv.reader(f)
        global soc
        soc = list(reader)
        soc = np.array(soc)
        soc = np.float64(soc)

    with open('./data/scdf.csv','r') as f:
        reader = csv.reader(f)
        global sch
        sch = list(reader)
        sch = np.array(sch)
        sch = np.float64(sch)

    with open('./data/topic1.csv','r', encoding='utf-8') as f:
        reader = csv.reader(f)
        global family
        family = list(reader)

    with open('./data/topic2.csv','r', encoding='utf-8') as f:
        reader = csv.reader(f)
        global school
        school = list(reader)

    with open('./data/topic3.csv','r', encoding='utf-8') as f:
        reader = csv.reader(f)
        global love
        love = list(reader)

    with open('./data/topic4.csv','r', encoding='utf-8') as f:
        reader = csv.reader(f)
        global society
        society = list(reader)

    with open('./data/word_vectors.csv', 'r', encoding='utf-8') as f:
        reader3 = csv.reader(f)
        global wordvec
        wordvec = list(reader3)
    
    wordmat = np.transpose(wordvec)
    global wordve
    wordve = list(wordmat[0])

def Djbamboo(x):
    sa = x

    pos_tagger = Twitter()

    def tokenize(doc):
        return ['/'.join(t) for t in pos_tagger.pos(str(doc), norm=True, stem=True)]

    docs = tokenize(sa)

    topic1 = set(["엄마/Noun","아빠/Noun","아버지/Noun","어머니/Noun","할머니/Noun","부모님/Noun",
                  "동생/Noun","가족/Noun","아들/Noun","집안/Noun","자식/Noun","결혼/Noun","이혼/Noun","사촌/Noun"])

    topic2 = set(["선배/Noun","새내기/Noun","후배/Noun","동기/Noun","동아리/Noun","행사/Noun","인사/Noun",
              "술자리/Noun","학생회/Noun","학교/Noun","학년/Noun","입학/Noun","활동/Noun","술/Noun"
              "개강/Noun","밥약/Noun","꼰대/Noun","존댓말/Noun","학번/Noun","학우/Noun","존댓말/Noun","학번/Noun","신입생/Noun"])

    topic3 = set(["사랑/Noun","마음/Noun","행복/Noun","감정/Noun","추억/Noun","상처/Noun","이별/Noun",
                "서로/Noun","연애/Noun","벚꽃/Noun","미안/Noun","후회/Noun","마지막/Noun",
                "소중/Noun","미소/Noun","표현/Noun","따뜻/Noun","첫사랑/Noun","웃음/Noun","곰신/Noun","고백/Noun",
                "성격/Noun","사이/Noun","서운/Noun","남자친구/Noun","여자친구/Noun"])

    topic4 = set(["사회/Noun","문제/Noun","여성/Noun","이유/Noun","의견/Noun","동성애/Noun","잘못/Noun","종교/Noun"
                "정치/Noun","집단/Noun","혐오/Noun","행위/Noun","차별/Noun","주장/Noun","가치관/Noun","정치/Noun","소수자/Noun",
                  "자유/Noun","발언/Noun"])


    words_count1 = 0
    words_count2 = 0
    words_count3 = 0
    words_count4 = 0
    for word in docs: 
        if word in topic1: 
            words_count1 += 1
        if word in topic2: 
            words_count2 += 1
        if word in topic3: 
            words_count3 += 1
        if word in topic4: 
            words_count4 += 1
    to = [words_count1,words_count2,words_count3,words_count4]
    max_index = to.index(max(to)) + 1


    def test(s):
        hangul = re.compile('[^ |가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자
        result = hangul.sub('', s) # 한글과 띄어쓰기를 제외한 모든 부분을 제거
        return(result)

    rex = test(str(docs))
    rex = rex.split(" ")

    ind = []
    for i in range(0,len(rex)-1):
        if rex[i] in wordve:
            ind.append(wordvec[wordve.index(rex[i])][1:])

    ind = np.array(ind)
    ind = ind.astype(np.float64)
    savec = sum(ind[0:len(ind)-1])/len(ind)

    def dot_product(v1, v2):
        return sum(v1*v2)
    def cosine_measure(v1, v2):
        return dot_product(v1, v2) / (math.sqrt(dot_product(v1, v1)) * math.sqrt(dot_product(v2, v2)))

    tem = []
    if(max_index==1):
        for i in range(1,len(fam)):
            try:
                temp = cosine_measure(savec,fam[i])
                tem.append(temp)
            except TypeError:
                tem.append(100)
    elif(max_index==2):
        for i in range(1,len(sch)):
            try:
                tem.append(cosine_measure(savec,sch[i]))
            except TypeError:
                tem.append(100)
    elif(max_index==3):
        for i in range(1,len(lov)):
            try:
                tem.append(cosine_measure(savec,lov[i]))
            except TypeError:
                tem.append(100)
    elif(max_index==4):
        for i in range(1,len(soc)):
            try:
                tem.append(cosine_measure(savec,soc[i]))
            except TypeError:
                tem.append(100)

        
    for i in range(len(tem)):
        if(~np.isnan(tem[i])==False):
            tem[i] = 0
        else:
            tem[i] = tem[i]


    reco1 = tem.index(heapq.nlargest(10,tem)[0])+1
    reco2 = tem.index(heapq.nlargest(10,tem)[1])+1
    reco3 = tem.index(heapq.nlargest(10,tem)[2])+1
    reco4 = tem.index(heapq.nlargest(10,tem)[3])+1
    reco5 = tem.index(heapq.nlargest(10,tem)[4])+1

    if(max_index==1):
        if(family[reco1][1:3] == family[reco2][1:3]):
            reco2 = reco4
            if(family[reco1][1:3] == family[reco3][1:3]):
                reco3 = reco5
        elif(family[reco3][1:3] == family[reco2][1:3]):
            reco2 = reco4
    elif(max_index==2):
        if(school[reco1][1:3] == school[reco2][1:3]):
            reco2 = reco4
            if(school[reco1][1:3] == school[reco3][1:3]):
                reco3 = reco5
        elif(school[reco3][1:3] == school[reco2][1:3]):
            reco2 = reco4
    elif(max_index==3):
        if(love[reco1][1:3] == love[reco2][1:3]):
            reco2 = reco4
            if(love[reco1][1:3] == love[reco3][1:3]):
                reco3 = reco5
        elif(love[reco3][1:3] == love[reco2][1:3]):
            reco2 = reco4
    elif(max_index==4):
        if(society[reco1][1:3] == society[reco2][1:3]):
            reco2 = reco4
            if(society[reco1][1:3] == society[reco3][1:3]):
                reco3 = reco5
        elif(society[reco3][1:3] == society[reco2][1:3]):
            reco2 = reco4

    if(max_index==1):
        song1 = family[reco1][1:3][0]
        name1 = family[reco1][1:3][1]
        song2 = family[reco2][1:3][0]
        name2 = family[reco2][1:3][1]
        song3 = family[reco3][1:3][0]
        name3 = family[reco3][1:3][1]
        # return (family[reco1][1:3],family[reco2][1:3],family[reco3][1:3])
    elif(max_index==2):
        song1 = school[reco1][1:3][0]
        name1 = school[reco1][1:3][1]
        song2 = school[reco2][1:3][0]
        name2 = school[reco2][1:3][1]
        song3 = school[reco3][1:3][0]
        name3 = school[reco3][1:3][1]
        # return (school[reco1][1:3],school[reco2][1:3],school[reco3][1:3])
    elif(max_index==3):
        song1 = love[reco1][1:3][0]
        name1 = love[reco1][1:3][1]
        song2 = love[reco2][1:3][0]
        name2 = love[reco2][1:3][1]
        song3 = love[reco3][1:3][0]
        name3 = love[reco3][1:3][1]
        # return (love[reco1][1:3],love[reco2][1:3],love[reco3][1:3])
    elif(max_index==4):
        song1 = society[reco1][1:3][0]
        name1 = society[reco1][1:3][1]
        song2 = society[reco2][1:3][0]
        name2 = society[reco2][1:3][1]
        song3 = society[reco3][1:3][0]
        name3 = society[reco3][1:3][1]
        # return (society[reco1][1:3],society[reco2][1:3],society[reco3][1:3])
    rst = {
        'song1':song1,
        'name1':name1,
        'song2':song2,
        'name2':name2,
        'song3':song3,
        'name3':name3,
    }
    return(rst)

comeondata()
# print (Djbamboo('당분간 널 찾지 않으려고 해 그동안은 정말 찾고싶었거든 인연을 가장해 우연인 척 널 보러 갔던 날 정말 말하고 싶었는데 너한테 연락 오고 하루도 빠짐없이 생각했다고 네가 누군가를 만난다는 사실에 밥을 먹다가 울어버렸다고 그리고 내가 정말 힘들 때 안기고 싶었던 품은 그 누구도 아닌 너였다고 있잖아 너한테 연락 왔을 때 나 사실 정말 많이 흔들렸다 너한테 그렇게 여름에 상처받고 바보같이 흔들렸어 진짜 바보같지 내가 물어봤었잖아 어떻게 하면 좋겠어 그리고 나 흔들리면 안 되겠지 너한테는 너무 어려운 질문이었을까 난 확신이 듣고 싶었던 건데 너한테 돌아오는 답변은 모르겠다 뿐이였지 그리고 네 말 날 너무 아프게했어 우린 이제 답이 없잖아 한때는 정말 말 예쁘게 하는 사람이라고 생각했던 너였는데 그런 네가 말로 나를 아프게 할 줄은 정말 꿈에도 몰랐거든 그리고 나 사실 굉장히 여려 그래서 아직도 네가 상처 줬던 말 품 안에 안고 살아 그래도 있잖아 학교생활이 너무 힘들어 눈물이 쏟아지고 사람들 시선이 너무 괴로워하지 말아야 할 생각이 드는 날이면 너부터 생각나 내가 지금 걷고 있는 이 거리에서 니 이름 부르면 딱 한 번만 뒤돌아줬으면 좋겠다 내가 지금 서 있는 이 공간에서 너 이름 부르면 딱 한 번만 달려 와줬으면 좋겠다 그리고 사실 요새는 정말 안 좋은 생각이 많이 들곤 하는데 나 한 번만 안아주면 안 될까 그러면 조금이라도 살고싶을텐데 진짜 보고 싶다 아직도 널 보고 싶어 하는 내가 너무 싫지만 왜 너는 싫지않은걸까 그리고 마지막으로 생일 축하했어 현재형으로 하고 싶은 말이었는데 또 과거형이네 이제 시간이 더 흐르게 된다면 나는 너에게 있어서 이제 과거도 아닌 아무것도 아닌 존재가 되어버리겠지 나에게 있어서 너는 늘 어려운 존재인데 어려운 존재 어려운 존재야 여기서는 말할 수있을것같아 정말 터무니없는 생각이고 헛된 꿈이라는 거 잘 아는데 너가 한 번만 와줬으면 좋겠다 다시 한 번만 딱 한 번만 말해줬으면 좋겠다 달보다 네가 더 예쁘다고'))
# print (Djbamboo('연대 숲 고민이 있어요. ㅠㅜㅠ 요즘 따라 연락을 하고 지내는 남자가 있어요. 원래 알던 사이이지만 방학을 한 거 나서 얼굴을 못 보게 되면서 카톡을 자주 하고 있거든요 밤에만 연락을 하게 돼요 바쁘긴 하지만 카톡을 전혀 오 가지 않고 저녁 이후부터 잠잘 때 까지만, 연락을 하고 있어요……. 카톡 내용은 뭔가…. 솔직히 남이 보면 호감이라고 느낄 것 같아요. 그렇지만 친구들 말에 의하면 저녁 6시 이후에 연락이 되는 남자는 그냥 외로워서 그런 거라나 맞는 말인 것 같아서 고민이 많네요!! 여러분도 밤에만 연락이 되는 남자는 제가 좋다기보다는 그냥 외로워서라고 생각하시나요!!?'))