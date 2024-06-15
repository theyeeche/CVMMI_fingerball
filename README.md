# 指動靈球
## Demo
### 畫面介紹
  - 左上角的數字: 分數
  - 橘框: 會移動，球遇到會觸發特定效果。
  - 綠框: 球遇到會增加分數
  - 紅框: 球遇到會減少分數
  - 球: 玩家利用手指控制

![image](https://github.com/theyeeche/CVMMI_fingerball/assets/61655288/c03bdcaa-1577-4a3c-8df2-31517b9925e6)

### 功能介紹

  - 若手掌不是處於張開動作時，球會沿著手指到球的圓心的設限方向做彈射。
  
  ![image](https://github.com/theyeeche/CVMMI_fingerball/assets/61655288/111d837f-aed2-4278-b40d-583556483767)
  
  - 若手掌處於張開的動作時，球會跑到中指下的關節處
  - 手掌回復關閉狀態時，球會固定在目前中指關節的位置。
  
  ![image](https://github.com/theyeeche/CVMMI_fingerball/assets/61655288/c279d502-d2a0-4e5d-a84f-894d24112b20)
  - 若球撞到橘框，會將紅框收起來，球會進行色彩的變換。

  ![image](https://github.com/theyeeche/CVMMI_fingerball/assets/61655288/b4a06084-a140-48fb-9a86-7077127b4376)
  
  
### Python Library Version Installation 
```
pip install opencv-python
pip install mediapipe
```
