#include "LLBC2.h"
#include "utils.h"
#include "correlationMeasures.h"
#include <assert.h>
#include <math.h>
#include <set>
#include <stdlib.h>
#include <iostream>
#include "incrementalLearner.h"

using namespace std;
LLBC2::LLBC2() :
trainingIsFinished_(false)
{
}
LLBC2::LLBC2(char* const *&, char* const *) :
dist(), trainingIsFinished_(false)
{
    name_ = "LLBC2";
}

LLBC2::~LLBC2()
{
    //析构函数
}
void LLBC2::reset(InstanceStream &is)
{
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();

    trainingIsFinished_ = false;

    for(CategoricalAttribute a = 0; a < 200;a++)
        for(CategoricalAttribute b = 0; b < 2;b++)
        {
            parents_[a][b] = NOPARENT;//对于父亲表进行一个初始化，其中a表示属性Xa,b代表第几个属性
        }

    dist.reset(is);

}
void LLBC2::getCapabilities(capabilities &c)
{
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void LLBC2::initialisePass()
{
    assert(trainingIsFinished_ == false);

}
void LLBC2::train(const instance &inst)
{
    dist.update(inst);
}

bool search_isExit1( int a[],int n, int x )
{
    for(int m= 0;m<n;m++)
    {
        if(x == a[m])
        {
            return true;
        }
    }
    return false;
}
void LLBC2::classify(const instance &inst, std::vector<double> &classDist)
{

    int  parentsLocal[200][2];
    for(CategoricalAttribute a = 0; a < 200;a++)
        for(CategoricalAttribute b = 0; b < 2;b++)
        {
            parentsLocal[a][b] = NOPARENT;
        }
    std::vector<float> mi_loc;
    getMutualInformationloc(dist.xxyCounts.xyCounts, mi_loc, inst);

    // calculate the conditional mutual information from the xxy distribution
    crosstab<float> cmi_loc = crosstab<float>(noCatAtts_);

    getCondMutualInfloc(dist.xxyCounts, cmi_loc, inst);

    crosstab<float> sum_loc = crosstab<float>(noCatAtts_);

    for(CategoricalAttribute i = 0; i < noCatAtts_; i++){
        for(CategoricalAttribute j = 0; j < noCatAtts_; j++){
            if(i == j){
                sum_loc[i][j] = -1000;
                continue;
            }
            sum_loc[i][j] = cmi_loc[i][j]+mi_loc[i];
        }
    }

    int attributeQueue_loc[noCatAtts_];
    for(int i = 0; i < noCatAtts_; i++){
        attributeQueue_loc[i] = -1;
    }

    int firstnode_loc = -1;
    float minmath = -1000;
    for(CategoricalAttribute i = 0 ; i < noCatAtts_;i++)
    {
        if(mi_loc[i] > minmath)
        {
            firstnode_loc = i;
            minmath = mi_loc[i];
        }
    }

    attributeQueue_loc[0] = firstnode_loc;
    parentsLocal[firstnode_loc][0] = -1;
    parentsLocal[firstnode_loc][1] = -1;


    int m = 1;
    int nextnode = -1;
    int father_loc1 = -1;
    while(m != noCatAtts_)
    {
        nextnode = -1;
        minmath = -10000;
        father_loc1 = -1;
        for(CategoricalAttribute a = 0 ; a < m ; a++)
        {
            for(CategoricalAttribute b = 0; b < noCatAtts_; b++)
            {
                if(search_isExit1(attributeQueue_loc,noCatAtts_,b))
                {
                    ;
                }else if(sum_loc[b][attributeQueue_loc[a]] > minmath)
                    {
                        father_loc1 = attributeQueue_loc[a];
                        nextnode = b;
                        minmath = sum_loc[b][attributeQueue_loc[a]];
                    }
            }

        }
        attributeQueue_loc[m] = nextnode;
        parentsLocal[nextnode][0] = father_loc1;
        m++;

    }
    //1-order LLBC is established

    int father_loc2 = -1;

    for(CategoricalAttribute a = 2 ; a < noCatAtts_ ; a++)
    {
        minmath = -10000;
        father_loc2 = -1;
        for(CategoricalAttribute b = 0; b < a; b++){
            if(parentsLocal[attributeQueue_loc[a]][0] == attributeQueue_loc[b]){
               ;
            }else if(cmi_loc[attributeQueue_loc[a]][attributeQueue_loc[b]] > minmath){
                father_loc2 = attributeQueue_loc[b];
                minmath = cmi_loc[attributeQueue_loc[a]][attributeQueue_loc[b]];
            }
        }
        parentsLocal[attributeQueue_loc[a]][1] = father_loc2;
    }

    std::vector<double> localPosteriorDist;
    localPosteriorDist.assign(noClasses_, 0);

    for (CatValue y = 0; y < noClasses_; y++)
    {
        localPosteriorDist[y] = dist.xxyCounts.xyCounts.p(y)* (std::numeric_limits<double>::max() / 2.0);
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++)
    {
        CategoricalAttribute parent ;
        if(parentsLocal[x1][1] !=NOPARENT)
        {
            parent = 2;
        }else if(parentsLocal[x1][1] == NOPARENT && parentsLocal[x1][0] != NOPARENT)
        {
            parent = 1;
        }else if(parentsLocal[x1][1] == NOPARENT && parentsLocal[x1][0] == NOPARENT)
        {
            parent = 0;
        }
        for (CatValue y = 0; y < noClasses_; y++)
        {
            if (parent == 0)
            {
                localPosteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate
            } else if (parent == 1)
            {
                const InstanceCount totalCount1 = dist.xxyCounts.xyCounts.getCount(parentsLocal[x1][0], inst.getCatVal(parentsLocal[x1][0]));
                if (totalCount1 == 0)
                {
                    localPosteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else
                {
                    localPosteriorDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parentsLocal[x1][0], inst.getCatVal(parentsLocal[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parent == 2)
            {
                const InstanceCount totalCount1 = dist.xxyCounts.getCount(parentsLocal[x1][0], inst.getCatVal(parentsLocal[x1][0]), parentsLocal[x1][1], inst.getCatVal(parentsLocal[x1][1]));
                if (totalCount1 == 0)
                {

                    const InstanceCount totalCount2 = dist.xxyCounts.xyCounts.getCount(parentsLocal[x1][0], inst.getCatVal(parentsLocal[x1][0]));
                    if (totalCount2 == 0)
                    {

                        localPosteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else
                    {
                        localPosteriorDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parentsLocal[x1][0], inst.getCatVal(parentsLocal[x1][0]), y);
                    }
                } else
                {   //p(x1=v1|Y=y, x2=v2, x3=v3) using M-estimate
                    localPosteriorDist[y] *= dist.p(x1, inst.getCatVal(x1), parentsLocal[x1][0], inst.getCatVal(parentsLocal[x1][0]), parentsLocal[x1][1], inst.getCatVal(parentsLocal[x1][1]), y);
                }
            }
        }
    }

    normalise(localPosteriorDist);

    //general data

    for (CatValue y = 0; y < noClasses_; y++)
    {
        classDist[y] = dist.xxyCounts.xyCounts.p(y)* (std::numeric_limits<double>::max() / 2.0);//类标签在实例中出现的次数
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++)
    {
        CategoricalAttribute parent ;
        if(parents_[x1][1] !=NOPARENT)
        {
            parent = 2;
        }else if(parents_[x1][1] == NOPARENT && parents_[x1][0] != NOPARENT)
        {
            parent = 1;
        }else if(parents_[x1][1] == NOPARENT && parents_[x1][0] == NOPARENT)
        {
            parent = 0;
        }
//        cout<<parent<<endl;
        for (CatValue y = 0; y < noClasses_; y++)
        {
            if (parent == 0)
            {
                classDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate
            } else if (parent == 1)
            {
                const InstanceCount totalCount1 = dist.xxyCounts.xyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]));
                if (totalCount1 == 0)
                {
                    classDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else
                {
                    classDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parent == 2)
            {
                const InstanceCount totalCount1 = dist.xxyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]), parents_[x1][1], inst.getCatVal(parents_[x1][1]));
                if (totalCount1 == 0)
                {

                    const InstanceCount totalCount2 = dist.xxyCounts.xyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]));
                    if (totalCount2 == 0)
                    {

                        classDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else
                    {
                        classDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), y);
                    }
                } else
                {   //p(x1=v1|Y=y, x2=v2, x3=v3) using M-estimate
                    classDist[y] *= dist.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), parents_[x1][1], inst.getCatVal(parents_[x1][1]), y);
                }
            }
        }
    }

    normalise(classDist);
    //ensemble learning
    for (int classno = 0; classno < noClasses_; classno++)
    {
        classDist[classno] += localPosteriorDist[classno];
        classDist[classno] = classDist[classno] / 2;
    }

}

void LLBC2::finalisePass()
{
    assert(trainingIsFinished_ == false);
    std::vector<float> mi;
    getMutualInformation(dist.xxyCounts.xyCounts, mi);

    crosstab<float> cmi = crosstab<float>(noCatAtts_);//

    getCondMutualInf(dist.xxyCounts, cmi);

    crosstab<float> sum = crosstab<float>(noCatAtts_);

    for(CategoricalAttribute i = 0; i < noCatAtts_; i++){
        for(CategoricalAttribute j = 0; j < noCatAtts_; j++){
            if(i == j){
                sum[i][j] = -1000;
                continue;
            }
            sum[i][j] = cmi[i][j]+mi[i];
        }
    }

    int attributeQueue[noCatAtts_];
    for(int i = 0; i < noCatAtts_; i++){
        attributeQueue[i] = -1;
    }

    int firstnode = -1;
    float minmath = -1000;
    for(CategoricalAttribute i = 0 ; i < noCatAtts_;i++)
    {
        if(mi[i] > minmath)
        {
            firstnode = i;
            minmath = mi[i];
        }
    }

    attributeQueue[0] = firstnode;
    parents_[firstnode][0] = -1;
    parents_[firstnode][1] = -1;


    int m = 1;
    int nextnode = -1;
    int father1 = -1;
    while(m != noCatAtts_)
    {
        nextnode = -1;
        minmath = -10000;
        father1 = -1;
        for(CategoricalAttribute a = 0 ; a < m ; a++)
        {
            for(CategoricalAttribute b = 0; b < noCatAtts_; b++)
            {
                if(search_isExit1(attributeQueue,noCatAtts_,b))
                {
                    ;
                }else if(sum[b][attributeQueue[a]] > minmath)
                    {
                        father1 = attributeQueue[a];
                        nextnode = b;
                        minmath = sum[b][attributeQueue[a]];
                    }
            }

        }
        attributeQueue[m] = nextnode;
        parents_[nextnode][0] = father1;
        m++;

    }


    int father2 = -1;

    for(CategoricalAttribute a = 2 ; a < noCatAtts_ ; a++)
    {
        minmath = -10000;
        father2 = -1;
        for(CategoricalAttribute b = 0; b < a; b++){
            if(parents_[attributeQueue[a]][0] == attributeQueue[b]){
               ;
            }else if(cmi[attributeQueue[a]][attributeQueue[b]] > minmath){
                father2 = attributeQueue[b];
                minmath = cmi[attributeQueue[a]][attributeQueue[b]];
            }
        }
        parents_[attributeQueue[a]][1] = father2;
    }


    trainingIsFinished_ = true;
}
bool LLBC2::trainingIsFinished()
{
    return trainingIsFinished_;
}
