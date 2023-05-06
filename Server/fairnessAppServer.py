import cherrypy
import pyximport
pyximport.install()
import io
import os, os.path
import random
import argparse
import socket
import html
import datetime
import operator
import csv
from collections import defaultdict, OrderedDict
import time
import pickle
import numpy as np
import heapq
from heapq import *
import fairnessMetrics
import probabilities
import fairnessTests
import copy
import cProfile
cp = cProfile.Profile()

theories = fairnessMetrics.fairness_theories
max_h_index = probabilities.max_h_index
items_truelabel_list = probabilities.items_truelabel_list

# default value of URL parameters
default_userid = "anonymous"
default_rgseed = None
default_rgseed_test = None
default_nprgseed = None
default_hypothesis = 2
default_timesteps = 20
default_port = 8080
default_uiversion = "v1"
default_swapchoice = "true"

# other defaults
default_return_code = int(round(time.time() * 10000))
default_items = np.array([[0, 1], [1, 1], [1, 1], [0, 0], [1, 0], [0, 0], [0, 1], [1, 0], [1, 0], [0, 0]])

class ec2(object):
    demographics_indices = [0, 1]
    dem_values = [[0, 1], [0, 1]]
    filled_test_queue = None
    precomputed_info = None
    rgstate_userid = None
    
    def __init__(self, userid=default_userid, timesteps=default_timesteps, hypothesis=default_hypothesis, 
        exhaustive=False, precompute=False, timestamp=None, checksubmoderr=False, rgseed_test=None, rgseed=None, nprgseed=None):
        self.observations = []
        self.t = 0
        if ec2.rgstate_userid == None:
            rgstate_saved = random.getstate()
            random.seed(100)
            ec2.rgstate_userid = random.getstate()
            random.setstate(rgstate_saved)
        if userid==default_userid:
            rgstate_saved = random.getstate()
            random.setstate(ec2.rgstate_userid)
            userid += str(random.randrange(10000000,99999999))
            ec2.rgstate_userid = random.getstate()
            random.setstate(rgstate_saved)
        self.userid = userid
        self.timesteps = max(1,timesteps)
        self.hypothesis = hypothesis
        self.exhaustive = exhaustive
        self.precompute = precompute
        self.checksubmoderr = checksubmoderr

        # prepapre the tests
        if ec2.filled_test_queue==None:
            ec2.filled_test_queue = fairnessTests.fill_test_queue(items_truelabel_list, ec2.demographics_indices, ec2.dem_values, rgseed=rgseed_test)
            if timestamp==None:
                timestamp = str(datetime.datetime.now().date()) + '-' + str(datetime.datetime.now().time()).replace(':', '_')
            f = open("logs/tests-"+timestamp+".txt", "w")
            for test in ec2.filled_test_queue:
                f.write("%s test_idx=%d tl=%s pl[0]=%s pl[1]=%s priors=%s\n"
                    %(test[1],test[2][0]['item_idx'],str(test[2][0]['tl']),str(test[2][0]['pl']),str(test[2][1]['pl']),str(test[3])))
            f.close()

        # initialize random number generators for this ec2 instance
        rgstate_saved = random.getstate()
        nprgstate_saved = np.random.get_state()
        random.seed(rgseed)
        np.random.seed(nprgseed)

        if precompute:
            if ec2.precomputed_info==None:
                print("Precomputing first iteration...")
                ts_start = time.time()
                ec2.precomputed_info = {"iter0info":{}, "iter1nfo":[None,None], "observations":[[],[]], "queue": [None,None]}
                ec2.precomputed_info = {"iter0info":{}, "iter1nfo":[None,None], "observations":[[],[]], "queue": [None,None]}
                ec2.precomputed_info["queue"] = [copy.deepcopy(ec2.filled_test_queue), copy.deepcopy(ec2.filled_test_queue)]
                ec2.precomputed_info["iter0info"]["test"] = [heappop(ec2.precomputed_info["queue"][0]),heappop(ec2.precomputed_info["queue"][1])][0]
                ec2.precomputed_info["iter0info"]["curr_Q_min"] = ec2.precomputed_info["queue"][0][0][0]
                for case_chosen in range(2):
                    ec2.precomputed_info["observations"][case_chosen].append([ec2.precomputed_info["iter0info"]["test"],case_chosen,dict()])
                    self.observations = ec2.precomputed_info["observations"][case_chosen]
                    if self.exhaustive:
                        min_priority, min_test, min_count, count, curr_Q_min = self.find_min_exhaustive(Q=ec2.precomputed_info["queue"][case_chosen])
                    else:
                        min_priority, min_test, min_count, count, curr_Q_min = self.find_min_submodular(Q=ec2.precomputed_info["queue"][case_chosen])
                    ec2.precomputed_info["iter1nfo"][case_chosen] = (min_priority, min_test, min_count, count, curr_Q_min)
                self.observations = []
                time_taken = time.time()-ts_start
                print("Precomputing DONE in time = ",time_taken)
            else:
                print("Will use precomputed first iteration...")
        else:
            self.Q = copy.deepcopy(ec2.filled_test_queue)

        self.rgstate = random.getstate()
        self.nprgstate = np.random.get_state()
        random.setstate(rgstate_saved)
        np.random.set_state(nprgstate_saved)

    def convert_list_crime(self, list_digits, predicted=True):
        answer = []
        zero_string = " will NOT Reoffend " if predicted else " did NOT Reoffend "
        one_string = " WILL Reoffend " if predicted else " DID Reoffend "
        for i in range(0, len(list_digits)):
            if (list_digits[i] == 0):
                answer.append("Person "+str(i)+": "+zero_string)
            else:
                answer.append("Person "+str(i)+": "+one_string)
        return answer

    def gen_prompt(self, test):
        return '{ "t":"%s", "p1":"%s", "p2":"%s", "itemfeatures":"%s", "t_n":%s, "p1_n":%s, "p2_n":%s, "itemfeatures_n":%s, "explanations":%s, "expval":%s}'%(
            str(self.convert_list_crime(test[2][0]["tl"], False)), 
            str(self.convert_list_crime(test[2][0]["pl"])), 
            str(self.convert_list_crime(test[2][1]["pl"])), 
            str(items_truelabel_list[test[2][0]["item_idx"]][0].tolist()),
            test[2][0]["tl"],
            test[2][0]["pl"],
            test[2][1]["pl"],
            str(items_truelabel_list[test[2][0]["item_idx"]][0].tolist()),
            str([self.gen_dropdown_stats(items_truelabel_list[test[2][0]["item_idx"]], test[2][0]["pl"]), self.gen_dropdown_stats(items_truelabel_list[test[2][0]["item_idx"]], test[2][1]["pl"])]),
            str(self.gen_dropdown_labels()).replace('\'', '"')
            )

    def gen_dropdown_stats(self, stats_items_truelabels, predicted_labels):
        stats_items = stats_items_truelabels[0]
        stats_tl = stats_items_truelabels[1]
        race_stats = []
        gender_stats = []
        intersection_stats = []
        all_stats_list = [[[0, 0, 0, 0],[0, 0, 0, 0]],[[0, 0, 0, 0],[0, 0, 0, 0]]]
        for curr_i_idx in range(len(stats_items)):
            curr_pl = predicted_labels[curr_i_idx]
            curr_tl = stats_tl[curr_i_idx]
            curr_gender = stats_items[curr_i_idx][0]
            curr_race = stats_items[curr_i_idx][1]
            if(curr_pl == 1):
               all_stats_list[curr_gender][curr_race][1] += 1 
            if(curr_tl == curr_pl):
                all_stats_list[curr_gender][curr_race][0] += 1
                if(curr_tl == 1):
                    all_stats_list[curr_gender][curr_race][2] += 1
                if(curr_pl == 1):
                    all_stats_list[curr_gender][curr_race][3] += 1
        for a_idx in range(0,4):
            race_stats.append([all_stats_list[0][0][a_idx]+all_stats_list[1][0][a_idx], all_stats_list[0][1][a_idx]+all_stats_list[1][1][a_idx]])
            gender_stats.append([all_stats_list[0][0][a_idx]+all_stats_list[0][1][a_idx], all_stats_list[1][0][a_idx]+all_stats_list[1][1][a_idx]])
            intersection_stats.append([all_stats_list[0][0][a_idx], all_stats_list[0][1][a_idx], all_stats_list[1][0][a_idx], all_stats_list[1][1][a_idx]])
        return [race_stats, gender_stats, intersection_stats]
        
    def gen_dropdown_labels(self):
        return [["B", "W"],["F", "M"], ["BF", "WF", "BM", "WM"]]

    def pick_case_by_hypothesis(self, items, noisy):
        if(self.hypothesis == 100):
            print("***random return selection***")
            return np.random.choice(np.arange(0,2))
        case_tuple = probabilities.priors(self.demographics_indices, self.dem_values, self.test, self.hypothesis)
        for h in range(0,max_h_index):
            loop_tuple = probabilities.priors(self.demographics_indices, self.dem_values, self.test, h)
            print(" Case 0 "+theories[h]+": "+str(loop_tuple[0])+", Case 1 "+theories[h]+": "+str(loop_tuple[1]))
        if noisy:
            case_chosen = np.random.choice(np.arange(0,2), p=[case_tuple[0], case_tuple[1]])
        else:
            case_chosen = 0 if case_tuple[0] > case_tuple[1] else 1
        print("Picked: "+str(case_chosen))
        return case_chosen

    def gen_user_response(self, noisy=False):
        prompt_string = "The True Labels are: "+str(self.test[2][0]["tl"])+\
                        ". \n Case 0 Predicted Labels are: "+str(self.test[2][0]["pl"])+". \n Case 1 Predicted Labels are: "+str(self.test[2][1]["pl"])+". \n" \
                        "Which Case is more fair? [Please return 0 or 1.] "
        print(prompt_string)
        items = items_truelabel_list[self.test[2][0]["item_idx"]][0].tolist()
        case_chosen = self.pick_case_by_hypothesis(items, noisy)
        return case_chosen

    # using exhaustive search find the entry in Q which has the smallest -objective and remove this entry
    def find_min_exhaustive(self, Q=None, old=False, debug=False, printq=False):
        if not Q: Q = self.Q
        count = 0
        min_priority = float("inf")
        min_test = None
        for i in range(len(Q)):
            candidate_test = Q[i]
            objective_value = -probabilities.test_objective(self.demographics_indices, self.dem_values, 
                self.observations, (-candidate_test[0], candidate_test[1], candidate_test[2], candidate_test[3]), 
                old=old, debug=debug)
            if printq:
                print("@@@","t:",self.t,"count:",count,"min:",min_priority,"id:",candidate_test[1],"old:",candidate_test[0],"new:",objective_value)
            try:
                assert (-objective_value)<=(-candidate_test[0])
            except:
                print("WARNING: Submodularity Violation:","count:",count,"id:",candidate_test[1],"old:",-candidate_test[0],
                    "new:",-objective_value,"max:",-min_priority)
            Q[i] = (objective_value, candidate_test[1], candidate_test[2], candidate_test[3])
            if min_priority > objective_value:
                min_priority = objective_value
                min_test = candidate_test
                min_count = count
            count += 1
        Q.remove((min_priority, min_test[1], min_test[2], min_test[3]))
        curr_Q_min = '<unknown>'
        return (min_priority, min_test, min_count, count, curr_Q_min)

    # making use of submodularity property find the entry in Q which has the smallest -objective and remove this entry
    # to do this, we organize Q as a heap and make use of summodularity property which states that the -objective
    # of a test can only go higher (i.e. objective can onnly go down) as time passes
    def find_min_submodular(self, Q=None, old=False, debug=False, printq=False):
        #if self.t==5:
        #   cp.enable()
        if not Q: Q = self.Q
        count = 0
        min_priority = float("inf")
        min_test = None
        curr_Q_min = Q[0][0]
        submoderr = False
        processed_ids = set()
        while min_priority > curr_Q_min:
            candidate_test = Q[0]
            processed_ids.add(candidate_test[1])
            objective_value = -probabilities.test_objective(self.demographics_indices, self.dem_values, 
                self.observations, (-candidate_test[0], candidate_test[1], candidate_test[2], candidate_test[3]), 
                old=old, debug=debug)
            if printq:
                print("@@@","t:",self.t,"count:",count,"min:",min_priority,"id:",candidate_test[1],"old:",candidate_test[0],"new:",objective_value)
            try:
                assert (-objective_value)<=(-candidate_test[0])
            except:
                submoderr = True
                print("WARNING: Submodularity Violation:","count:",count,"id:",candidate_test[1],"old:",-candidate_test[0],
                    "new:",-objective_value,"max:",-min_priority)
            #heappush(Q, (objective_value, test[1], test[2]))
            #print(min_priority,candidate_test[0],objective_value)
            heapq.heapreplace(Q, (objective_value, candidate_test[1], candidate_test[2], candidate_test[3]))
            if min_priority > objective_value:
                min_priority = objective_value
                min_test = candidate_test
                min_count = count
            curr_Q_min = Q[0][0]
            count += 1
        if self.checksubmoderr and submoderr:
            print("Submodularity Violation: checking if it caused problem.")
            real_min_priority = min_priority
            real_min_test = min_test
            violation_count = 0
            for i in range(len(Q)):
                candidate_test = Q[i]
                if not(candidate_test[1] in processed_ids):
                    objective_value = -probabilities.test_objective(self.demographics_indices, self.dem_values, 
                        self.observations, (-candidate_test[0], candidate_test[1], candidate_test[2], candidate_test[3]), 
                        old=old, debug=debug)
                    if (-objective_value)>(-candidate_test[0]):
                        violation_count += 1
                    if real_min_priority > objective_value:
                        real_min_priority = objective_value
                        real_min_test = candidate_test
            if real_min_priority<min_priority:
                print("WARNING: Submodularity Violation resulted in incorrect Argmax. Real:: ",
                    "id:",real_min_test[1], "old:",-real_min_test[0],"new:",-real_min_priority)
            else:
                print("NOTE: Luckily submodularity Violation had no adverse impact.")
            print("NOTE: there exist %d additional Submodularity Violations"%(violation_count))

        Q.remove((min_priority, min_test[1], min_test[2], min_test[3]))
        heapq.heapify(Q)
        return (min_priority, min_test, min_count, count, curr_Q_min)
    
    def print_probs(self):
        print("Probabilities after t = %d are:"%(self.t-1))
        probs = []
        for h in range(0, max_h_index):
            p = probabilities.p_h_observations(self.demographics_indices, self.dem_values, self.observations, h)
            print(" Hypothesis " + str(theories[h]) + " : " + str(p))
            probs.append(p)
        return probs

    def sample(self, case_chosen=None, reason1=None, reason2=None, reasonff=None, debug=False, old=False, 
        printq=False, timestamp=None, csvwriter=None):
        ts_start = time.time()
        if not timestamp:
            timestamp = time.strftime("%Y%m%d%H%M%S%Z", time.gmtime())
        
        rgstate_saved = random.getstate()
        nprgstate_saved = np.random.get_state()
        random.setstate(self.rgstate)
        np.random.set_state(self.nprgstate)


        if (self.t == 0):
            count = 0
            if self.precompute:
                print("\n**** t = %d with queue length=%d choice = %s "%(self.t,len(self.filled_test_queue),case_chosen))
                print("userid:", self.userid, "| debug:", debug, "| old:", old)
                print("using precomputed values")
                self.test = copy.deepcopy(self.precomputed_info["iter0info"]["test"])
                curr_Q_min = copy.deepcopy(self.precomputed_info["iter0info"]["curr_Q_min"])
            else:
                print("\n**** t = %d with queue length=%d choice = %s "%(self.t,len(self.Q),case_chosen))
                print("userid:", self.userid, "| debug:", debug, "| old:", old)
                self.test = heappop(self.Q)
                curr_Q_min = self.Q[0][0]
            prompt_json = self.gen_prompt(self.test)
            min_priority = -self.test[0]
            min_id = self.test[1]
            time_taken = time.time()-ts_start
            print("Time taken = ",time_taken)
            print("Iteration count = ",count)
            print("+++ ","t:",self.t,"| len(Q):",len(self.filled_test_queue),"| count:",count,"| time_taken:",time_taken,
                "| curr_Q_min:",curr_Q_min,"| min_priority:",min_priority,"| min_count: ",0,"| min_id:",min_id)
            probs = [""]*4
        else:
            self.observations.append([self.test,case_chosen,dict()])
            probs = self.print_probs()
            if self.t == self.timesteps:
                self.test = ("","","")
                prompt_json = "Done"
            elif (probabilities.p_observations(self.demographics_indices, self.dem_values, self.observations) == 0):
                self.test = ("","","")
                prompt_json = "Done"
            elif self.t==1 and self.precompute:
                print("\n**** t = %d with queue length=%d choice = %s "%(self.t,1+len(self.precomputed_info["queue"][case_chosen]),case_chosen))
                print("userid:", self.userid, "| debug:", debug, "| old:", old)
                print("using precomputed values")
                min_priority, min_test, min_count, count, curr_Q_min = copy.deepcopy(self.precomputed_info["iter1nfo"][case_chosen])
                self.Q = copy.deepcopy(self.precomputed_info["queue"][case_chosen])
                self.test = (-min_priority, min_test[1], min_test[2], min_test[3])
                prompt_json = self.gen_prompt(self.test)
                time_taken = time.time()-ts_start
                print("Time taken = ",time_taken)
                print("Iteration count = ",count)
                print("+++ ","t:",self.t,"| len(Q):",len(self.Q)+1,"| count:",count,"| time_taken:",time_taken,
                    "| curr_Q_min:",curr_Q_min,"| min_priority:",min_priority,"| min_count:",min_count,"| min_id:",min_test[1])
            else:
                print("\n**** t = %d with queue length=%d choice = %s "%(self.t,len(self.Q),case_chosen))
                print("userid:", self.userid, "| debug:", debug, "| old:", old)
                if self.exhaustive:
                    min_priority, min_test, min_count, count, curr_Q_min = self.find_min_exhaustive(old=old, debug=debug, printq=printq)
                else:
                    min_priority, min_test, min_count, count, curr_Q_min = self.find_min_submodular(old=old, debug=debug, printq=printq)
                self.test = (-min_priority, min_test[1], min_test[2], min_test[3])
                prompt_json = self.gen_prompt(self.test)
                time_taken = time.time()-ts_start
                print("Time taken = ",time_taken)
                print("Iteration count = ",count)
                print("+++ ","t:",self.t,"| len(Q):",len(self.Q)+1,"| count:",count,"| time_taken:",time_taken,
                    "| curr_Q_min:",curr_Q_min,"| min_priority:",min_priority,"| min_count:",min_count,"| min_id:",min_test[1])
        if csvwriter:
            csvwriter.writerow([timestamp, "sample", self.userid, "", "", "", "", "", "", "", "",
                case_chosen, reason1, reason2, self.t, self.test[1], self.test[0], probs[0], probs[1], probs[2], probs[3], reasonff])
        self.t += 1

        self.rgstate = random.getstate()
        self.nprgstate = np.random.get_state()
        random.setstate(rgstate_saved)
        np.random.set_state(nprgstate_saved)

        return prompt_json


class FairnessExperimenter(object):
    def __init__(self, InfoLogFile, precompute=False, checksubmoderr=False, exhaustive=False): 
        self.IndexTemplate = {'value':None, 'timestamp':None}
        self.InfoLogFile = InfoLogFile
        self.InfoLogWriter = csv.writer(InfoLogFile)
        self.precompute = precompute
        self.checksubmoderr = checksubmoderr
        self.exhaustive = exhaustive
        self.InfoLogWriter.writerow(["timestamp", "method", "userid", "hypothesis", 
            "timesteps", "rgseed", "nprgseed", "debug", "old", "exhaustive", "uiversion",
            "choice", "reason1", "reason2", "t", "testid", "value", "prob_a", "prob_fd", "prob_fn", "prob_dp", "reasonff"])

    @cherrypy.expose
    def index(self,userid=default_userid,hypothesis=default_hypothesis,timesteps=default_timesteps,
        rgseed=default_rgseed,nprgseed=default_nprgseed,swapchoice=default_swapchoice,
        debug="false",old="false",exhaustive="false",checksubmoderr="false",uiversion=default_uiversion):
        timestamp = time.strftime("%Y%m%d%H%M%S%Z", time.gmtime())
        cherrypy.session['debug'] = True if debug.lower()!="false" else False
        cherrypy.session['old'] = True if old.lower()!="false" else False
        checksubmoderr = True if checksubmoderr.lower()!="false" else self.checksubmoderr
        exhaustive = True if exhaustive.lower()!="false" else self.exhaustive
        cherrypy.session['return_code'] = str(int(round(time.time() * 10000)))
        cherrypy.session['uiversion'] = str(uiversion)
        if (swapchoice!="false"):
            swapchoice="true"
        if rgseed!=None:
            rgseed = int(rgseed)
        if nprgseed!=None:
            nprgseed = int(nprgseed)
        hypothesis = int(hypothesis)
        timesteps=int(timesteps)

        cherrypy.session['ec2object'] = ec2(userid=userid,hypothesis=hypothesis,timesteps=timesteps,exhaustive=exhaustive,
            precompute=self.precompute,checksubmoderr=checksubmoderr,rgseed=rgseed,nprgseed=nprgseed)
        index_template_file = 'templates/index.html'
        index_template_file_mtime = os.path.getmtime(index_template_file)
        if not self.IndexTemplate['value'] or self.IndexTemplate['timestamp']<index_template_file_mtime:
            with open(index_template_file, 'r') as f:
                self.IndexTemplate['timestamp']=index_template_file_mtime
                self.IndexTemplate['value']=f.read()
        response = self.IndexTemplate['value']
        response = response.replace("{{userid}}",html.escape(cherrypy.session['ec2object'].userid))
        response = response.replace("{{return_code}}",html.escape(cherrypy.session['return_code']))
        response = response.replace("{{uiversion}}",html.escape(cherrypy.session['uiversion']))
        response = response.replace("{{numtests}}",html.escape(str(timesteps)))
        response = response.replace("{{swapchoice}}",html.escape(swapchoice))
        self.InfoLogWriter.writerow([timestamp, "index", cherrypy.session['ec2object'].userid, hypothesis, timesteps, rgseed, nprgseed,
            debug, old, exhaustive, uiversion, "", "", "", "", "", "", "", "", ""])
        return response

    @cherrypy.expose
    def sample(self, choice=None, reason1=None, reason2=None, reasonff=None):
        timestamp = time.strftime("%Y%m%d%H%M%S%Z", time.gmtime())
        try:
            debug = cherrypy.session['debug']
        except:
            debug = False
        try:
            old = cherrypy.session['old']
        except:
            old = False
        if choice!=None:
            choice = int(choice)
        # now get the sample
        sample = cherrypy.session['ec2object'].sample(choice,reason1=reason1,reason2=reason2,reasonff=reasonff,
            debug=debug,old=old,timestamp=timestamp,csvwriter=self.InfoLogWriter)
        #sample = '{ "t":"0 1 1", "p1":"1 0 0", "p2":"1 1 1"}'
        self.InfoLogFile.flush()
        return(sample)

def CORS(): 
    cherrypy.response.headers["Access-Control-Allow-Origin"] = "*" 

config = {
    'global': {
        'server.socket_host':'127.0.0.1',
        'server.socket_port': default_port,
        'log.error_file' : 'logs/Web.log',
        'log.access_file' : 'logs/Access.log'
    },
    '/': {
        'tools.CORS.on': True,
        'tools.sessions.on': True,
        'tools.staticdir.root': os.path.abspath(os.getcwd()),
    },
    '/static': {
        'tools.staticdir.on': True,
        'tools.staticdir.dir': './public'
    }
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Server for EC2 Fairness webapp.')
    parser.add_argument('--local', dest='localhost', action='store_const',
        const=True, default=False,
        help='use localhost for interactive mode (default: default IP address)')
    parser.add_argument('--port', dest='port', 
        type=int, nargs=1, default=default_port,
        help='TCP port to listen on (default: %d)'%(default_port))
    parser.add_argument('--testmode', dest='testmode', action='store_const',
        const=True, default=False,
        help='use test mode with no web server (default: False)')
    parser.add_argument('--debug', dest='debug', action='store_const',
        const=True, default=False,
        help='use debug (default: False)')
    parser.add_argument('--old', dest='old', action='store_const',
        const=True, default=False,
        help='use old (default: False)')
    parser.add_argument('--exhaustive', dest='exhaustive', action='store_const',
        const=True, default=False,
        help='use exhaustive search (default: False)')
    parser.add_argument('--noisy', dest='noisy', action='store_const',
        const=True, default=False,
        help='use noisy (default: False)')
    parser.add_argument('--usegini', dest='usegini', action='store_const',
        const=True, default=False,
        help='use gini (default: False)')
    parser.add_argument('--printq', dest='printq', action='store_const',
        const=True, default=False,
        help='print queue (default: False)')
    parser.add_argument('--checksubmoderr', dest='checksubmoderr', action='store_const',
        const=True, default=False,
        help='check if submod error had impact (default: False)')
    parser.add_argument('--precompute', dest='precompute', action='store_const',
        const=True, default=False,
        help='precompute first iteration (default: False)')
    parser.add_argument('--hypothesis', dest='hypothesis', 
        type=int, nargs=1, default=default_hypothesis,
        help='hypothesis id in test mode (default: %d)'%(default_hypothesis))
    parser.add_argument('--timesteps', dest='timesteps', 
        type=int, nargs=1, default=default_timesteps,
        help='# of timesteps in test mode (default: %d)'%(default_timesteps))
    parser.add_argument('--rgseed', dest='rgseed', 
        type=int, nargs=1, default=default_rgseed,
        help='seed for random numbers (default: %s)'%(str(default_rgseed)))
    parser.add_argument('--rgseed_test', dest='rgseed_test', 
        type=int, nargs=1, default=None,
        help='seed for random numbers for test generation (default: %s)'%(str(default_rgseed_test)))
    parser.add_argument('--nprgseed', dest='nprgseed', 
        type=int, nargs=1, default=default_nprgseed,
        help='seed for numpy random numbers (default: %s)'%(str(default_nprgseed)))
    args = parser.parse_args()
    if args.rgseed!=default_rgseed:
        args.rgseed = args.rgseed[0]
    if args.rgseed_test!=default_rgseed_test:
        args.rgseed_test = args.rgseed_test[0]
    if args.nprgseed!=default_nprgseed:
        args.nprgseed = args.nprgseed[0]
    if args.timesteps!=default_timesteps:
        args.timesteps = args.timesteps[0]
    if args.hypothesis!=default_hypothesis:
        args.hypothesis = args.hypothesis[0]
    if args.port!=default_port:
        args.port = args.port[0]
    timestamp = str(datetime.datetime.now().date()) + '-' + str(datetime.datetime.now().time()).replace(':', '_')
    print("timestamp:", timestamp)
    print('testmode:',args.testmode, '| debug:',args.debug, '| old:', args.old, '| exhaustive:', args.exhaustive,
        '| noisy:', args.noisy, '| usegini:', args.usegini, '| printq:', args.printq, '| checksubmoderr:', args.checksubmoderr, 
        '| precompute:', args.precompute, '| hypothesis:', args.hypothesis, '| timesteps:', args.timesteps, 
        '| rgseed:', args.rgseed, '| rgseed_test:', args.rgseed_test, '| nprgseed:',args.nprgseed)

    if args.testmode:
        print("Running in non-interactive test mode")
        random.seed(args.rgseed)
        np.random.seed(args.nprgseed)
        probabilities.usegini = args.usegini
        ec2object = ec2(hypothesis=int(args.hypothesis), timesteps=int(args.timesteps), exhaustive=args.exhaustive, 
            precompute=args.precompute, timestamp=timestamp, checksubmoderr=args.checksubmoderr, 
            rgseed_test=args.rgseed_test, rgseed=args.rgseed, nprgseed=args.nprgseed)
        response = ec2object.sample(case_chosen=None, debug=args.debug, old=args.old, printq=args.printq)
        while response!="Done":
            case_chosen = ec2object.gen_user_response(noisy=args.noisy)
            response = ec2object.sample(case_chosen=case_chosen, debug=args.debug, old=args.old, printq=args.printq)
    else:
        if args.localhost:
            config['global']['server.socket_host'] = '127.0.0.1'
        else:
            config['global']['server.socket_host'] = socket.gethostbyname(socket.gethostname())
        config['global']['server.socket_port'] = args.port
        random.seed(args.rgseed)
        np.random.seed(args.nprgseed)
        probabilities.usegini = args.usegini
        # precomputation
        ec2(hypothesis=int(args.hypothesis), timesteps=int(args.timesteps), exhaustive=args.exhaustive, 
            precompute=args.precompute, timestamp=timestamp, rgseed_test=args.rgseed_test)
        print("Listening on IP address ",config['global']['server.socket_host']," on port ",config['global']['server.socket_port'])
        InfoLogFilename = "logs/info-"+timestamp+".csv"
        cherrypy.tools.CORS = cherrypy.Tool('before_finalize', CORS)
        with open(InfoLogFilename,'w', newline='') as infologfile:
            cherrypy.quickstart(FairnessExperimenter(infologfile, precompute=args.precompute, checksubmoderr=args.checksubmoderr, 
                exhaustive=args.exhaustive), config=config)
