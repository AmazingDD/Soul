import os
import csv
import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ['PowerLogger', 'getDevice', 'printFullReport']

"""
[before monitoring]

    pc.printFullReport(pc.getDevice())
    pl = pc.PowerLogger(interval=0.05)
    pl.start()
    time.sleep(5)
    pl.recordEvent(name='Process Start')

[monitoring]
[other code project]

    time.sleep(5)
    pl.stop()
    filename = logfilename.split("/")[-1]
    pl.showDataTraces(filename=filename)
    print(str(pl.eventLog))
    pc.printFullReport(pc.getDevice())

[after monitoring]

"""

device_nodes = {
    'jetson_tx2':[
        ('module/main' , '0041', '0'),
        ('module/gpu'  , '0040', '0'),
        ('module/ddr'  , '0041', '2'),
        ('module/cpu'  , '0041', '1'),
        ('module/soc'  , '0040', '1'),
        ('module/wifi' , '0040', '2'),
 
        ('board/main'        , '0042', '0'),
        ('board/5v0-io-sys'  , '0042', '1'),
        ('board/3v3-sys'     , '0042', '2'),
        ('board/3v3-io-sleep', '0043', '0'),
        ('board/1v8-io'      , '0043', '1'),
        ('board/3v3-m.2'     , '0043', '2'),
    ],
    'jetson_nx':[
        ('main', '0040','1'),
        ('cpu+gpu', '0040','2'),
        ('soc', '0040','3')
    ],
    'jetson_agx_orin': [
        ('gpu_soc', '0040', '1'),  # VDD_GPU_SOC
        ('cpu_cv', '0040', '2'),  # VDD_CPU_CV
        ('sys_5v0', '0040', '3'),  # VIN_SYS_5V0
        ('vddq_vdd2_1v8', '0041', '2'), # VDDQ_VDD2_1V8AO
    ],
    'jetson_orin_nx': [
        ('vdd_in', '0040', '1'),  # vdd_in
        ('vdd_cpu_gpu_cv', '0040', '2'),  # vdd_cpu_gpu_cv
        ('vdd_soc', '0040', '3'),  # vdd_soc
    ]
}

driver_dir = {  
    'jetson_tx2':'/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/',
    'jetson_nx': '/sys/bus/i2c/devices/7-0040/',
    'jetson_agx_orin': '/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon2/',
    'jetson_orin_nx': '/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/',
}

_valTypes = ['power', 'voltage', 'current']
_valTypesFull = ['power [mW]', 'voltage [mV]', 'current [mA]']

def getNodes(device='jetson_agx_orin'):
    """Returns a list of all power measurement nodes, each a 
    tuple of format (name, i2d-addr, channel)"""
    assert device in device_nodes
    return device_nodes[device]

def powerSensorsPresent(device='jetson_agx_orin'):
    """Check whether we are on the TX2 platform/whether the sensors are present"""
    return os.path.isdir(driver_dir[device])

def getDevice():
    for dir in driver_dir:
        if powerSensorsPresent(dir): 
            return dir 
        
def getPowerMode():
    return os.popen("nvpmodel -q | grep 'Power Mode'").read()[15:-1]

def readValue(i2cAddr='0040', channel='0', valType='power',device='jetson_agx_orin'):
    """Reads a single value from the sensor"""
    if device == 'jetson_tx2':
        fname = '/sys/bus/i2c/drivers/ina3221x/0-%s/iio:device%s/in_%s%s_input' % (i2cAddr, i2cAddr[-1], valType, channel)
        f = open(fname, 'r')
        res = f.read()
        f.close()
        return res  
    elif device == 'jetson_nx':
        res = {}
        for valtype in ['voltage','current']:
            val = 'in' if valtype=='voltage' else 'curr'
            # fname = '/sys/class/hwmon/hwmon4/%s%s_input' % (val,channel)
            fname='/sys/bus/i2c/drivers/ina3221/7-0040/hwmon/hwmon4/%s%s_input'% (val,channel)
            f = open(fname, 'r')
            res[valtype]  = f.read()
            f.close()
        res['power'] = eval(res['voltage'])*eval(res['current'])/1000
        return res[valType]
    elif device == 'jetson_agx_orin':
        res = {}
        for valtype in ['voltage', 'current']:
            val = 'in' if valtype == 'voltage' else 'curr'
            # Construct the filename for Jetson AGX Orin device
            fname = f'/sys/bus/i2c/drivers/ina3221/1-{i2cAddr}/hwmon/hwmon1/{val}{channel}_input'
            if i2cAddr == '0041':
                fname = f'/sys/bus/i2c/drivers/ina3221/1-{i2cAddr}/hwmon/hwmon2/{val}{channel}_input'
            with open(fname, 'r') as f:
                res[valtype] = f.read()
        res['power'] = eval(res['voltage'])*eval(res['current'])/1000
        return res[valType]
    elif device == 'jetson_orin_nx': 
        res = {}
        for valtype in ['voltage', 'current']:
            val = 'in' if valtype == 'voltage' else 'curr'
            # Construct the filename for Jetson AGX Orin device
            fname = f'/sys/bus/i2c/drivers/ina3221/1-{i2cAddr}/hwmon/hwmon3/{val}{channel}_input'
            with open(fname, 'r') as f:
                res[valtype] = f.read()
        
        # Calculate power if needed
        res['power'] = float(res['voltage']) * float(res['current']) / 1000
        return res[valType]
    
def getModulePower():
    """Returns the current power consumption of the entire module in mW."""
    return float(readValue(i2cAddr='0041', channel='0', valType='power'))

def getAllValues(nodes,device):
    """Returns all values (power, voltage, current) for a specific set of nodes."""
    return [[float(readValue(i2cAddr=node[1], channel=node[2], valType=valType, device=device)) for valType in _valTypes] for node in nodes]

def printFullReport(device='jetson_agx_orin'):
    """Prints a full report, i.e. (power,voltage,current) for all measurement nodes."""
    header = []
    header.append('A description')
    for vt in _valTypesFull:
        header.append(vt)

    resultTable = []
    for descr, i2dAddr, channel in device_nodes[device]:
        row = []
        row.append(descr)
        for valType in _valTypes:
            row.append(readValue(i2cAddr=i2dAddr, channel=channel, valType=valType,device=device))
        resultTable.append(row)

    total = {}
    for i, vt in enumerate(header):
        if i<2: total.update({vt:[row[i] for row in resultTable]})
        else: total.update({vt:[eval(row[i]) for row in resultTable]})

    totalTable = pd.DataFrame(total)
    print(totalTable)

class PowerLogger:
    """This is an asynchronous power logger. 
    Logging can be controlled using start(), stop(). 
    Special events can be marked using recordEvent(). 
    Results can be accessed through 
    """
    def __init__(self, interval=0.01, figsave=False, csvwrite=False):
        """
        Constructs the power logger and sets a sampling interval (default: 0.01s) 
        and fixes which nodes are sampled (default: all of them)
        """
        self.interval = interval
        self._startTime = -1
        self.eventLog = []
        self.dataLog = []
        self.figsave = figsave
        self.csvwrite = csvwrite
        self.device = getDevice()
        self._nodes = device_nodes[self.device]

    def start(self):
        "Starts the logging activity"""
        #define the inner function called regularly by the thread to log the data
        def threadFun():
            #start next timer
            self.start()
            #log data
            t = self._getTime() - self._startTime
            self.dataLog.append((t, getAllValues(self._nodes,self.device)))
            #ensure long enough sampling interval
            t2 = self._getTime() - self._startTime
            # assert(t2-t < self.interval)
             
        #setup the timer and launch it
        self._tmr = threading.Timer(self.interval, threadFun)
        self._tmr.start()
        if self._startTime < 0:
            self._startTime = self._getTime()

    def _getTime(self):
        return time.clock_gettime(time.CLOCK_REALTIME)
    
    def recordEvent(self, name):
        """Records a marker a specific event (with name)"""
        t = self._getTime() - self._startTime
        self.eventLog.append((t, name))
 
    def stop(self):
        """Stops the logging activity"""
        self._tmr.cancel()

    def getDataTrace(self, nodeName='', valType='power'):
        # if getDevice() == 'jetson_nx': nodeName = 'main'
        """Return a list of sample values and time stamps for a specific measurement node and type"""
        pwrVals = [itm[1][[n[0] for n in self._nodes].index(nodeName)][_valTypes.index(valType)] 
                    for itm in self.dataLog]
        timeVals = [itm[0] for itm in self.dataLog]
        return timeVals, pwrVals
    
    def showDataTraces(self, names=None, valType='power', showEvents=True, filename='folder_name'):
        device = getDevice()
        if device == 'jetson_tx2':
            if names == None: 
                names = [name for name, _, _ in [self._nodes[i] for i in [0,1,2,3,11]]]
                label_names = ['System-wise', 'GPU', 'RAM', 'CPU', 'SSD']
                styles = [':', '-', '-', '-', '-', '--']
            else: 
                names = [name for name, _, _ in [self._nodes[i] for i in names]]
                label_names = names
                half = int((len(names)-1)/2)
                styles = [":"] + ['-'] * half + ['-.'] * (len(names) - half) + ['--']
        elif device == 'jetson_nx':
            if names == None: 
                names = [name for name, _, _ in [node for node in self._nodes]]
                label_names = ['System-wise', 'GPU+CPU+CV', 'SOC']
                styles = [':', '-', '--']
            else: 
                names = [name for name, _, _ in [self._nodes[i] for i in names]]
                label_names = names
                half = int((len(names)-1)/2)
                styles = [":"] + ['-'] * half + ['-.'] * (len(names) - half) + ['--']
        elif device == 'jetson_agx_orin': 
            if names is None: 
                names = [name for name, _, _ in [node for node in self._nodes]]
                label_names = ['GPU+SOC', 'CPU+CV', 'I/O (5V0)','I/O (VDDQ)']
                styles = ['-', '-', '-.', ':']
            else: 
                names = [name for name, _, _ in [self._nodes[i] for i in names]]
                label_names = names
                half = int((len(names)-1)/2)
                styles = [":"] + ['-'] * half + ['-.'] * (len(names) - half) + ['--']
        elif device == 'jetson_orin_nx': 
            if names is None: 
                names = [name for name, _, _ in [node for node in self._nodes]]
                label_names = ['vdd_in', 'vdd_cpu_gpu_cv', 'vdd_soc']
                styles = ['-', '-.', ':']
            else: 
                names = [name for name, _, _ in [self._nodes[i] for i in names]]
                label_names = names
                half = int((len(names)-1)/2)
                styles = [":"] + ['-'] * half + ['-.'] * (len(names) - half) + ['--']

        #prepare data to display
        TPs = [self.getDataTrace(nodeName=name, valType=valType) for name in names]
        Ts, _ = TPs[0] # TPs.shape 4 2 344
        Ps = [p for _, p in TPs] # Ps.shape 4 344

        if device == 'jetson_tx2':
            Ps.append([Ps[1][i]+Ps[2][i]+Ps[3][i] for i in range(len(Ts))])
        elif device == 'jetson_nx':
            Ps.append([Ps[0][i]+Ps[1][i]+Ps[2][i] for i in range(len(Ts))])
        elif device == 'jetson_agx_orin': 
            Ps.append([Ps[0][i]+Ps[1][i]+Ps[2][i]+Ps[3][i] for i in range(len(Ts))])
        elif device == 'jetson_orin_nx': 
            Ps.append([Ps[0][i]+Ps[1][i]+Ps[2][i] for i in range(len(Ts))])

        Ts_s = []
        Ps_a = []
        Ps_t = []
        for i in range(int(len(Ts)/5)):
            Ts_s.append(Ts[(5*i)+2])
        for p in range(len(Ps)):
            Ps_ap = []
            Ps_tp = []
            for i in range(int(len(Ts)/5)):
                part = Ps[p][(5*i):(5*i)+5]
                Ps_ap.append(np.mean(part))
                Ps_tp.append(np.max(part))
            Ps_a.append(Ps_ap)
            Ps_t.append(Ps_tp)

        # nodeNames=[nodeName for nodeName in names] #['gpu_soc', 'cpu_cv', 'sys_5v0', 'vddq_vdd2_1v8']
        # print(nodeNames)
        energies = [self.getTotalEnergy(nodeName=nodeName) for nodeName in names]
        if device == 'jetson_tx2':
            energies.append(energies[1]+energies[2]+energies[3])
        elif device == 'jetson_nx':
            energies.append(energies[0]+energies[1]+energies[2])
        elif device == 'jetson_agx_orin':
            energies.append(energies[0]+energies[1]+energies[2]+energies[3])
        elif device == 'jetson_orin_nx':
            energies.append(energies[0]+energies[1]+energies[2])
        
        for t in range(len(label_names)): 
            print(f'{label_names[t]}: {energies[t] / 1e3:.4f} J')
        print(f'Sum: {np.sum(energies) / 1e3:.4f} J')

        os.makedirs(f'./energy_logs/{filename}', exist_ok=True)

        if self.figsave:
            for t in range(len(label_names)): 
                plt.plot(Ts, Ps[t], label=f'{label_names[t]} ({energies[t] / 1e3:.4f} J)', linestyle=styles[t])
                print(f'{label_names[t]} ({energies[t] / 1e3:.4f} J)')

            plt.xlabel('time [s]')
            plt.ylabel(_valTypesFull[_valTypes.index(valType)])
            plt.grid(True)
            ln = plt.legend(loc='center left', bbox_to_anchor=(1.04,0.5))

            plt.title('%s trace (NVPModel: %s)' % (valType, os.popen("nvpmodel -q | grep 'Power Mode'").read()[15:-1]))

            if showEvents:
                for t, _ in self.eventLog:
                    plt.axvline(x=t, color='black', linestyle='-.')

            plt.savefig(
                f'energy_logs/{filename}/{valType}_total.png', 
                bbox_extra_artists=(ln,), 
                bbox_inches='tight'
            )
            plt.close()

        if self.csvwrite: 
            with open(f'energy_logs/{filename}/{valType}.csv', 'w') as f:
                csvf = csv.writer(f)
                csvf.writerow(['time'] + label_names)
                for i in range(len(Ts)):
                    csvf.writerow([Ts[i]] + [Ps[j][i] for j in range(len(label_names))])
                csvf.writerow([0] + energies)

    def getTotalEnergy(self, nodeName='', valType='power'):
        """Integrate the power consumption over time."""
        timeVals, dataVals = self.getDataTrace(nodeName=nodeName, valType=valType)
        assert(len(timeVals) == len(dataVals))
        tPrev, wgtdSum = 0.0, 0.0
        for t, d in zip(timeVals, dataVals):
            wgtdSum += d*(t-tPrev)
            tPrev = t
        return wgtdSum
    
    def getAveragePower(self, nodeName='', valType='power'):
        energy = self.getTotalEnergy(nodeName=nodeName, valType=valType)
        timeVals, _ = self.getDataTrace(nodeName=nodeName, valType=valType)
        return energy/timeVals[-1]
    
if __name__ == "__main__":
    device = getDevice()
    print(f"Detected device: {device}")  # 打印出设备名称
    printFullReport(getDevice())

    pl = PowerLogger(interval=0.05)
    pl.start()
    time.sleep(5)
    print('5s IDLE time passed, start IO bench mark now!')
    pl.recordEvent('started IO bench mark')
    time.sleep(2)
    pl.recordEvent('ding! 3s')
    os.system('stress -c 12 -t 3')
    time.sleep(1.5)
    pl.recordEvent('ding! 2s')
    os.system('stress -c 1 -t 2')
    time.sleep(2)
    pl.recordEvent('ding! 1s')
    os.system('stress -c 2 -t 1 -m 4')
    time.sleep(1.5)

    pl.stop()
    pl.showDataTraces()
    nodename = device_nodes[pl.device][2][0]
    pl.showMostCommonPowerValue(nodename)

    '''
    printFullReport(getDevice())
    pl = PowerLogger(interval=0.05)
    pl.start()
    time.sleep(5)
    pl.recordEvent(name='Process Start')

    # [Your Code for testing]

    time.sleep(5)
    pl.stop()
    filename = f'./'
    pl.showDataTraces(filename=filename)
    print(str(pl.eventLog))
    printFullReport(getDevice())

    '''
