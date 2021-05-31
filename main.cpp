#include <iostream>
#include <string>

/* 
链表节点的定义，每个节点包括当前人的id，密码和指向下一个人的指针
*/
struct ListNode
{
	int id;
	int password;
	ListNode* next;

	ListNode(const int id_, const int password_, ListNode* next_ = nullptr) {
		id = id_;
		password = password_;
		next = next_;
	}
};

/* 
顺序表的定义，包括当前在队伍内的人的id和相应的密码，以及队伍人数
*/
struct SqList
{
	int* id;
	int* passwords;
	int size;
	int current = 0;

	SqList(int size_, int* passwords_) {
		size = size_;
		passwords = passwords_;
		id = new int[size];
		for (int i = 0; i < size; i++) {
			id[i] = i + 1;
		}
	}
};

SqList* InitSqList(int n, int* passwords) {
	SqList* p = new SqList(n, passwords);
	return p;
}


ListNode* InitListNode(int n, int* passwords) {
	ListNode* tail, * newNode = nullptr;
	newNode = new ListNode(n, passwords[n - 1], newNode);
	tail = newNode;
	if (n > 1) {
		for (int i = n - 2; i >= 0; i--) {
			newNode = new ListNode(i + 1, passwords[i], newNode);
		}
	}
	//使链表成环
	tail->next = newNode;
	return newNode;
}

void printIds(ListNode* head, int m) {
	using namespace std;
	//如果队伍只有一人，则输出此人的id，并结束约瑟夫环过程
	if (head->next == head) {
		cout << (head->id) << endl;
		delete head;
		return;
	}

	ListNode* p, * q;
	if (m == 1) {
		//如果当前人要出列，则要遍历列表一圈找到上一个人
		p = head;
		q = head;
		while (q->next != p) q = q->next;
	}
	else {
		p = head->next;
		q = head;
		m -= 1;
		for (int i = 1; i < m; i++) {
			q = q->next;
			p = p->next;
		}
	}
	cout << p->id << endl;
	m = p->password;
	q->next = p->next;
	delete p;
	//对剩下的链表递归调用该函数
	printIds(q->next, m);
}

void printIds(SqList* p, int m) {
	using namespace std;
	if (p->size == 1) {
		//如果队伍只剩一人，则直接输出id
		cout << p->id[p->size - 1] << endl;
		delete p;
		return;
	}
	//通过取模的方式实现从顺序表的末尾跳到开头
	int new_current = (p->current + m - 1) % p->size;

	cout << p->id[new_current] << endl;

	int new_m = (p->passwords)[new_current];
	//将出列人后面的人的id和密码往前移动一格
	for (int i = new_current; i < p->size - 1; i++) {
		p->id[i] = p->id[i + 1];
		p->passwords[i] = p->passwords[i + 1];
	}
	p->size--;
	p->current = new_current;
	printIds(p, new_m);
}

int main(int argc, char** argv) {
	using namespace std;
	string method;
	cout << "please input method: LinkedList or SqList?" << endl;
	getline(cin, method);
	while (method != "LinkedList" && method != "SqList") {
		cin.clear();
		cin.sync();
		cout << "Wrong input! Please input again: " << endl;
		getline(cin, method);
	}
	int n = 0;

	cout << "please input n: " << endl;
	while (!(cin >> n) || n <= 0) {
		cin.clear();
		cin.sync();
		cout << "n must be a integer and greater than 0, please input again: " << endl;
	}
	cout << "please input n passwords: " << endl;
	int* arr_password = new int[n];
	for (int i = 0; i < n; i++) {
		cin >> arr_password[i];
		if (arr_password[i] <= 0) {
			cout << "Wrong input! passwords must be greater than 0, please restart!";
			return 0;
		}
	}

	int m = 0;
	cout << "please input the first m: " << endl;
	while (!(cin >> m) || m <= 0) {
		cin.clear();
		cin.sync();
		cout << "The first m must be a integer and greater than 0, please input again: " << endl;
	}

	cout << "The answer is:" << endl;
	if (method == "LinkedList") {
		ListNode* head = InitListNode(n, arr_password);
		printIds(head, m);
	}
	else {
		SqList* sqlist = InitSqList(n, arr_password);
		printIds(sqlist, m);
	}
	return 0;
}