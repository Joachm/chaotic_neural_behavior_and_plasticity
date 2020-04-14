from snnClass import *
import matplotlib.pyplot as plt
import seaborn as sns
tf.enable_eager_execution()
sns.set_style('darkgrid')

trials =1
dead = 0
excHist = []
inhHist = []
plastic = False
visualize = (False, 'exc')


for trial in range(trials):

    eRec = snnLayer('E', 100., 400, 1.,-60., -40., -40., 0.7, 0.03, -2, -50., 100. )
    iRec = snnLayer('I', 20., 100, -1.,  -55., -40., -40., 1, 0.2, 0.025, -55., 100. )

    eRec.addRecUniCon(0.000, 0.1, 1, 0.25)
    eRec.addUniCon('iRec', iRec.size, 0.0001, 0.1, 0.5, 0.25 )

    iRec.addRecUniCon(0.000, 0.1, 0.5, 0.25)
    iRec.addUniCon('eRec', eRec.size, 0.0001, 0.1, 0.5, 0.8 )


    epN = eRec.size*0.1
    ipN = iRec.size*0.1

    step = 1
    exc = 500/step
    iexc = 100/5/step
    inh = 400/step
    inh2 = 200/step

    '''
    exc = 20000/epN/step
    iexc = 600/ipN/step
    inh = 29/step
    '''

    #np.random.seed(0)

    pct = 0.06
    inp1 =np.random.choice([0,2000], size= (eRec.size, 1), p=[1-pct, pct])
    inp2 = np.random.choice([0,200], size= (iRec.size, 1), p=[1-pct, pct])



    eRec.updateNeurons(inp1, step)
    iRec.updateNeurons(inp2, step)

    eRec.reset()
    iRec.reset()

    actE = np.sum(eRec.active.numpy())

    actI = np.sum(iRec.active.numpy())

    numTimeSteps = 1000
    #plt.ion()
    for i in range(numTimeSteps):
        if i%200 == 0:
            print(i, '/', numTimeSteps)
        toE = eRec.sendOutput('recurrent', exc) + iRec.sendOutput('eRec', inh)
        toI = eRec.sendOutput('iRec', iexc) + iRec.sendOutput('recurrent', inh2)

        eRec.updateNeurons(toE, step)
        iRec.updateNeurons(toI, step)

        if plastic == True:
            #'''
            
            eRec.recurrentSTDP(0.9, 1.01)
            eRec.STDP('iRec', iRec.active, 0.9, 1.01)
            
            iRec.recurrentiSTDP(0.9, 1.05)
            iRec.STDP('eRec', eRec.active, 0.9, 1.05)
            #'''
            

        if visualize[0] == True:
            if visualize[1] == 'exc':
                sns.heatmap(eRec.active.numpy().reshape(20,20))
            else:
                sns.heatmap(iRec.active.numpy().reshape(10,10))
            plt.title(str(trial)+str(i) + '\n active exc: '+ str(np.sum(eRec.active.numpy())) + '\n active inh: ' + str(np.sum(iRec.active.numpy()))  )
     
            plt.pause(0.25)
            plt.clf()
        excHist.append(np.sum(eRec.active.numpy()))
        inhHist.append(np.sum(iRec.active.numpy()))
        

        eRec.reset()
        iRec.reset()

        actE = np.sum(eRec.active.numpy())
        #print(actE)

        actI = np.sum(iRec.active.numpy())
        #print(actI)
        #print()
        #'''
       
        '''
        if actE == 0 and actI==0:
            eRec.updateNeurons(inp1, step)
            iRec.updateNeurons(inp2, step)

            eRec.reset()
            iRec.reset()
        '''
        if actE + actI ==0:
            break
    if actE ==0:
        dead +=1
    
plt.close()
plt.clf()
plt.plot(excHist, label='exc')
plt.plot(inhHist, label='inh')
plt.legend()
if plastic == True:
    plt.title('Network activity with plasticity')
else:
    plt.title('Network activity witout plasticity')
plt.xlabel('time step')
plt.ylabel('number of spiking neurons')
plt.show()
plt.savefig('NetworkActivityPlasticity_'+str(plastic)+'.jpg')




