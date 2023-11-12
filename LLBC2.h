#include "incrementalLearner.h"
#include "learner.h"
#include "xxxyDist.h"
#include "xxyDist.h"
#include "xyDist.h"
#include "crosstab3d.h"
#include <limits>
class LLBC2 : public IncrementalLearner
{
    public:
        LLBC2();
        LLBC2(char* const *& argv, char* const * end);
        ~LLBC2(void);
        void reset(InstanceStream &is);   ///< reset the learner prior to training
        void initialisePass(); ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
        void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
        void finalisePass(); ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
        bool trainingIsFinished(); ///< true iff no more passes are required. updated by finalisePass()
        void getCapabilities(capabilities &c);

        virtual void classify(const instance &inst, std::vector<double> &classDist);
    protected:

    private:
        unsigned int noCatAtts_;          ///< the number of categorical attributes.
        unsigned int noClasses_;                          ///< the number of classes

        InstanceStream* instanceStream_;
        int parents_[200][2];//
        xxxyDist dist;//


        bool trainingIsFinished_; ///< true iff the learner is trained

        const static CategoricalAttribute NOPARENT = 0xFFFFFFFFUL;
};

