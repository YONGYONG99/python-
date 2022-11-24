# SVM으로 이미지 분류
# 세계 정치인 중 일부 사진을 사용

from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.pipeline import make_pipeline


faces = fetch_lfw_people(min_faces_per_person = 60, color=False) # 흑백사진
print(faces)
# 탐색기에 C:\Users\acorn\scikit_learn_data\lfw_home 가보기
print(faces.DESCR)
print('엥')
print(faces.data[:3], ' ', faces.data.shape)
print('오')
print(faces.target, set(faces.target))
print('캬')
print(faces.target_names)
print(faces.images.shape) # (1348, 62, 47) 62행 47열에다가 벡터값을 넣어놓은게 1348개 있다.

# print(faces.images[0]) # 이미지에 0번째 사람
# print(faces.target_names[faces.target[0]])
# plt.imshow(faces.images[0], cmap='bone')
# plt.show()

'''
fig, ax = plt.subplots(3, 5)
# print(fig)
# print(ax.flat) # 틀만들기
# plt.show()
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks = [], yticks=[], xlabel=faces.target_names[faces.target[i]])
plt.show()
'''

# 주성분 분석으로 이미지 차원 축소          # (1348, 62, 47)
m_pca = PCA(n_components = 150, whiten=True, random_state=0)
x_low = m_pca.fit_transform(faces.data)
print('x_low : ', x_low[:1], x_low.shape) # (1348, 150)
print(m_pca.explained_variance_ratio_)

# model
m_svc = SVC(C=1)
model = make_pipeline(m_pca, m_svc) # 선처리기(주성분 분석)와 분류기를 하나의 pipeline으로 묶어 순차적으로 진행
print(model)
# Pipeline(steps=[('pca', PCA(n_components=150, random_state=0, whiten=True)), ('svc', SVC(C=1))])

# train / test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(faces.data, faces.target, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (1011, 2914) (337, 2914) (1011,) (337,)

model.fit(x_train, y_train)
pred = model.predict(x_test)
print('에측값 : ',pred[:10])
print('실제값 : ',y_test[:10])
                
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print('classification_report : \n', classification_report(y_test, pred, target_names=faces.target_names))
con_mat = confusion_matrix(y_test, pred)
print('con_mat : \n', con_mat)
# con_mat : 
#  [[  7   0   0   7   0   0   0   0]
#  [  0  46   0   8   0   0   0   0]
#  [  0   3  15  12   0   0   0   0]
#  [  0   2   0 132   0   0   0   0]
#  [  0   0   0   8  22   0   0   1]
#  [  0   2   0   9   0   9   0   0]
#  [  0   1   0   6   0   0   5   0]
#  [  0   0   0   9   1   0   0  32]]
# 대각값들이 맞게 분류한것 
print('acc : ', accuracy_score(y_test, pred)) # acc :  0.7952522255192879

fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(x_test[i].reshape(62,47), cmap='bone')
    axi.set(xticks = [], yticks=[])
    axi.set_ylabel(faces.target_names[pred[i]].split()[-1],
                   color='black' if pred[i] == y_test[i] else 'red')
    fig.suptitle('pred result', size=14)
plt.show()













