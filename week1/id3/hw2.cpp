#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>


#include <utility>
#include <list>
#include <vector>
#include <string>
#include <set>
#include <map>


#include <cctype>
#include <cmath>
#include <cassert>

using namespace std;

#define Type int   //样本数据类型
 
#define   Map1        std::map< int, Type >    //定义一维map
#define   Map2        std::map< int, Map1 >    //定义二维map
#define   Map3        std::map< int, Map2 >    //定义三维map
#define   Pair        std::pair<int, Type>
#define   List        std::list< Pair >        //一维list
#define   SampleSpace std::list< List >        //二维list 用于存放样本数据
#define   Child       std::map< int, Node* >   //定义后继节点集合
#define   CI          const_iterator
 
/*
 *   在ID3算法中，用二维链表存放样本，结构为list< list< pair<int, int> > >，简记为SampleSpace，取名样本空间
 *   样本数据从根节点开始往下遍历。每一个节点的定义如下结构体
 */
 
struct Node
{
    int index;                    //当前节点样本最大增益对应第index个属性，根据这个进行分类的
    int type;                     //当前节点的类型
    Child next;                   //当前节点的后继节点集合
    SampleSpace sample;           //未分类的样本集合
};


typedef map<string,int>::iterator msiit;
typedef map<int,string>::iterator misit;
typedef map<int,int>::iterator miiit;


struct Data{

    int catlog;
    vector<int> data;

    int getData(int pos)
    {
        return data.at(pos);
    }

    void addFeature(int num)
    {
        data.push_back(num);
    }

    void setCatlog(int c)
    {
        catlog = c;
    }

    int getCatlog()
    {
        return catlog;
    }

    void Print(){
        cout << "Data " << this << endl;
        for(int i = 0; i < data.size(); ++i)
            cout << data[i] << " ";
        cout << "-- cat: " << catlog << endl;
    }

    void Print(map<int,string> &mp){
        cout << endl << "Data " << this << endl;
        for(int i = 0; i < data.size(); ++i)
            cout << mp[data[i]] << " ";
        cout << endl << "-- cat: " << mp[catlog] << endl;
    }
};

class ID3{
 
public:
 
    ID3(int );    
    ~ID3();
 
    void PushData(const Type*, const Type);   //将样本数据Push给二维链表
    void PushData(const std::vector<int>&, const int);
    void Build();                             //构建决策树
    int  Match(const Type*);                  //根据新的样本预测结果
    int  Match(const std::vector<int>&);
    void Print();                             //打印决策树的节点的值
 
private:
 
    void   _clear(Node*);
    void   _build(Node*, int);
    int    _match(const int*, Node*);
    void   _work(Node*);
    double _entropy(const Map1&, double);
    int    _get_max_gain(const SampleSpace&);
    void   _split(Node*, int);
    void   _get_data(const SampleSpace&, Map1&, Map2&, Map3&);
    double _info_gain(Map1&, Map2&, double, double);
    int    _same_class(const SampleSpace&);
    void   _print(Node*);
 
private:
 
    int dimension;
    Node *root;
};
 
//初始化ID3的数据成员
ID3::ID3(int dimension)
{
    this->dimension = dimension;
 
    root = new Node();
    root->index = -1;
    root->type = -1;
    root->next.clear();
    root->sample.clear();
}
 
//清空整个决策树
ID3::~ID3()
{
    this->dimension = 0;
    _clear(root);
}
 
//x为dimension维的属性向量，y为向量x对应的值
void ID3::PushData(const Type *x, const Type y)
{
    List single;
    single.clear();
    for(int i = 0; i < dimension; i++)
        single.push_back(make_pair(i + 1, x[i]));
    single.push_back(make_pair(0, y));
    root->sample.push_back(single);
}
void ID3::PushData(const vector<int> &x, const int y)
{
    List single;
    single.clear();
    for(int i = 0; i < dimension; ++i)
        single.push_back(make_pair(i + 1, x[i]));
    single.push_back(make_pair(0, y));
    root->sample.push_back(single);
}
 
void ID3::_clear(Node *node)
{
    Child &next = node->next;
    Child::iterator it;
    for(it = next.begin(); it != next.end(); ++it)
        _clear(it->second);
    next.clear();
    delete node;
}
 
void ID3::Build()
{
    _build(root, dimension);
}
 
void ID3::_build(Node *node, int dimension)
{
    //获取当前节点未分类的样本数据
    SampleSpace &sample = node->sample;
 
    //判断当前所有样本是否是同一类，如果不是则返回-1
    int y = _same_class(sample);
 
    //如果所有样本是属于同一类
    if(y >= 0)
    {
        node->index = -1;
        node->type = y;
        return;
    }
 
    //在_max_gain()函数中计算出当前节点的最大增益对应的属性，并根据这个属性对数据进行划分
    _work(node);
 
    //Split完成后清空当前节点的所有数据，以免占用太多内存
    sample.clear();
 
    Child &next = node->next;
    for(Child::iterator it = next.begin(); it != next.end(); ++it)
        _build(it->second, dimension - 1);
}
 
//判断当前所有样本是否是同一类，如果不是则返回-1
int ID3::_same_class(const SampleSpace &ss)
{
    //取出当前样本数据的一个Sample
    const List &f = ss.front();
 
    //如果没有x属性，而只有y，直接返回y
    if(f.size() == 1)
        return f.front().second;
 
    Type y = 0;
    //取出第一个样本数据y的结果值
    for(List::CI it = f.begin(); it != f.end(); ++it)
    {
        if(!it->first)
        {
            y = it->second;
            break;
        }
    }
 
    //接下来进行判断，因为list是有序的，所以从前往后遍历，发现有一对不一样，则所有样本不是同一类
    for(SampleSpace::CI it = ss.begin(); it != ss.end(); ++it)
    {
        const List &single = *it;
        for(List::CI i = single.begin(); i != single.end(); ++i)
        {
            if(!i->first)
            {
                if(y != i->second)
                    return -1;         //发现不是同一类则返回-1
                else
                    break;
            }
        }
    }
    return y;     //比较完所有样本的输出值y后，发现是同一类，返回y值。
}
 
void ID3::_work(Node *node)
{
    int mai = _get_max_gain(node->sample);
    assert(mai >= 0);
    node->index = mai;
    _split(node, mai);
}
 
//获取最大的信息增益对应的属性
int ID3::_get_max_gain(const SampleSpace &ss)
{
    Map1 y;
    Map2 x;
    Map3 xy;
 
    _get_data(ss, y, x, xy);
    double s = ss.size();
    double entropy = _entropy(y, s);   //计算熵值
 
    int mai = -1;
    double mag = -1;
 
    for(Map2::iterator it = x.begin(); it != x.end(); ++it)
    {
        double g = _info_gain(it->second, xy[it->first], s, entropy);    //计算信息增益值
        if(g > mag)
        {
            mag = g;
            mai = it->first;
        }
    }
 
    if(!x.size() && !xy.size() && y.size())   //如果只有y数据
        return 0;
    return mai;
}
 
//获取数据，提取出所有样本的y值，x[]属性值，以及属性值和结果值xy。
void ID3::_get_data(const SampleSpace &ss, Map1 &y, Map2 &x, Map3 &xy)
{
    for(SampleSpace::CI it = ss.begin(); it != ss.end(); ++it)
    {
    int c = 0;
        const List &v = *it;
        for(List::CI p = v.begin(); p != v.end(); ++p)
        {
            if(!p->first)
            {
                c = p->second;
                break;
            }
        }
        ++y[c];
        for(List::CI p = v.begin(); p != v.end(); ++p)
        {
            if(p->first)
            {
                ++x[p->first][p->second];
                ++xy[p->first][p->second][c];
            }
        }
    }
}
 
//计算熵值
double ID3::_entropy(const Map1 &x, double s)
{
    double ans = 0;
    for(Map1::CI it = x.begin(); it != x.end(); ++it)
    {
        double t = it->second / s;
        ans += t * log2(t);
    }
    return -ans;
}
 
//计算信息增益
double ID3::_info_gain(Map1 &att_val, Map2 &val_cls, double s, double entropy)
{
    double gain = entropy;
    for(Map1::CI it = att_val.begin(); it != att_val.end(); ++it)
    {
        double r = it->second / s;
        double e = _entropy(val_cls[it->first], it->second);
        gain -= r * e;
    }
    return gain;
}
 
//对当前节点的sample进行划分
void ID3::_split(Node *node, int idx)
{
    Child &next = node->next;
    SampleSpace &sample = node->sample;
 
    for(SampleSpace::iterator it = sample.begin(); it != sample.end(); ++it)
    {
        List &v = *it;
        for(List::iterator p = v.begin(); p != v.end(); ++p)
        {
            if(p->first == idx)
            {
                Node *tmp = next[p->second];
                if(!tmp)
                {
                    tmp = new Node();
                    tmp->index = -1;
                    tmp->type = -1;
                    next[p->second] = tmp;
                }
                v.erase(p);
                tmp->sample.push_back(v);
                break;
            }
        }
    }
}
 
int ID3::Match(const Type *x)
{
    return _match(x, root);
}  

int ID3::Match(const std::vector<int> &x){
    int *v = new int[sizeof(int) * x.size()];
    for(int i = 0; i < x.size(); ++i)
        v[i] = x[i];
    int ans = _match(v, root);
    delete[] v;
    return ans;
}

int ID3::_match(const Type *v, Node *node)
{
    if(node->index < 0)
        return node->type;
 
    Child &next = node->next;
    Child::iterator p = next.find(v[node->index - 1]);
    if(p == next.end()){
        if(p == next.begin())
            return -1;
        --p;
        return _match(v, p->second);
    }
 
    return _match(v, p->second);
}
 
void ID3::Print()
{
    _print(root);
}
 
void ID3::_print(Node *node)
{
    cout << "Index    = " << node->index << endl;
    cout << "Type     = " << node->type << endl;
    cout << "NextSize = " << node->next.size() << endl;
    cout << endl;
 
    Child &next = node->next;
    Child::iterator p;
    for(p = next.begin(); p != next.end(); ++p)
        _print(p->second);
}

void read(ifstream &fin){
    cout << "Please input dataset name" << endl;
    
    string dataFile;
    cin >> dataFile;
    
    fin.open(dataFile.c_str());
    if(!fin.is_open()){
        cout << "Open DataSet Error!" << endl;
        exit(-1);
    }
}
void sequenceString(ifstream &fin, map<string,int> &str2id, map<int,string> &id2str){//特征文本排序
    fin.clear();
    fin.seekg(0, ios::beg);
    //Reset IO Stream

    string Line;
    stringstream tmpStream;
    
    while(!fin.eof()){
        getline(fin, Line);

        for(int i = 0; i < Line.length(); ++i)
            if(Line[i] == ',' || Line[i] == '.')
                Line[i] = ' '; //替换非英文数字为空格自动分词
        tmpStream.clear();
        tmpStream << Line;
        //用字符串流分割单词

        string tmp;
        while(tmpStream){
            tmpStream >> tmp;
            if(tmp.length() == 0 && !isalpha(tmp[0]) && !isdigit(tmp[0]))
                continue; //必须是合法单词
            
            str2id[tmp] = 0; //在map中新建单词
            //cout << "Word :" << tmp << endl;
        }
    }
    str2id["Unknow"] = 0; //加入空向量标志

    int ct = 1;
    for(msiit it = str2id.begin(); it != str2id.end(); ++it){
        it->second = ct;
        id2str[ct] = it->first;
        ++ct;
        cout << "Word " << it->first << " id = " << it->second << endl;
    }//遍历所有单词，构造id一一对应
}
int makeFeatureVector(ifstream &fin, map<string,int> &str2id, map<int,string> &id2str, vector<Data*> &dataVec){
    fin.clear();
    fin.seekg(0, fin.beg);

    string Line;
    stringstream tmpStream;
    
    getline(fin, Line);
    cout << Line << endl;
    
    int FeatureNumber = 0;
    for(int i = 0; i < Line.length(); ++i)
        if(Line[i] == ',')
            FeatureNumber++;
    cout << "Feature Number = " << FeatureNumber << endl;
    //根据第一行统计向量长度

    while(!fin.eof()){
        getline(fin, Line);
        if(Line.length() <= 1)
            continue; //无效行

        Data* vec = new Data();
        dataVec.push_back(vec);
        //新建样本vector

        for(int i = 0; i < Line.length(); ++i)
            if(Line[i] == ',' || Line[i] == '.')Line[i] = ' '; //替换,为空格自动分词
        tmpStream.clear();
        tmpStream << Line;
        
        int Fcnt = 0;
        string tmp;
        
        #ifdef DBG
        cout << "---NEW LINE---" << endl;
        #endif
        
        while(tmpStream){
            tmpStream >> tmp;
            if(tmp.length() == 0 && !isalpha(tmp[0]) && !isdigit(tmp[0]))
                continue; //必须是合法单词
            
            if(str2id.count(tmp) == 0)
                tmp = "Unknow";
            
            #ifdef DBG
            cout << "|" << tmp ;
            #endif
            
            if(Fcnt == FeatureNumber){//读入分类
                vec->setCatlog(str2id[tmp]);
                
                #ifdef DBG
                vec->Print(id2str);
                #endif
                
                break;
            }else{//读入特征
                vec->addFeature(str2id[tmp]);
                Fcnt++;
            }
        }
        #ifdef DBG
        cout << endl << "---END LINE---" << endl;
        #endif
    }
    return FeatureNumber;
}

int main(){
    map<string,int>  str2id;
    map<int,string>  id2str;
    vector<Data*>    allData;
    vector<Data*>    testData;
    ifstream fin;


    read(fin);
    sequenceString(fin, str2id, id2str);
    int FeatureNumber = makeFeatureVector(fin, str2id, id2str, allData);
    //预处理训练数据

    cout << endl << "Pushing Data Vector..." << endl;
    
    ID3 Tree(FeatureNumber);
    for(int i = 0; i < allData.size(); ++i)
        Tree.PushData(allData[i]->data, allData[i]->getCatlog());
    
    cout << "Building Tree..." << endl;
    
    Tree.Build();
    
    #ifdef DBG
    Tree.Print();
    #endif

    cout << allData.size() << " Train Data" << endl;
    fin.close();//关闭训练集文件
    
    //读入待预测数据
    read(fin);
    makeFeatureVector(fin, str2id, id2str, testData);
    cout << testData.size() << " Test Data" << endl;

    int correct = 0, wrong = 0;
    for(int i = 0; i < testData.size(); ++i){
        int ans = Tree.Match(testData[i]->data);//预测
        
        cout << "Predict " << i << "'s data is ";
        cout << id2str[ans] << "  Ans = " << id2str[testData[i]->getCatlog()] << endl;
        if(ans == testData[i]->getCatlog())
            ++correct;
        else 
            ++wrong; //统计正确率
    }
    cout << setiosflags(ios::fixed) << setprecision(2);
    cout << "Correct Rate " << (100.0 * correct / (correct + wrong)) << '%' << endl;
    return 0;
}

