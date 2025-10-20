#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>

namespace VirtualAd {
namespace Integration {

/**
 * @brief スレッドセーフなキュー
 * 
 * @tparam T キューに格納する要素の型
 * 
 * このクラスは、複数のスレッドから安全にアクセスできるキューを提供します。
 * push()とpop()操作はミューテックスで保護され、キューが満杯または空の場合は
 * 条件変数を使用して待機します。タイムアウト機能もサポートしています。
 * 
 * 使用例:
 * @code
 * ThreadSafeQueue<int> queue(10);  // 最大10要素
 * 
 * // プロデューサースレッド
 * if (queue.push(42, 1000)) {
 *     std::cout << "Push succeeded" << std::endl;
 * }
 * 
 * // コンシューマースレッド
 * int value;
 * if (queue.pop(value, 1000)) {
 *     std::cout << "Popped: " << value << std::endl;
 * }
 * @endcode
 */
template<typename T>
class ThreadSafeQueue {
public:
    /**
     * @brief コンストラクタ
     * 
     * @param max_size キューの最大サイズ（デフォルト: 10）
     */
    explicit ThreadSafeQueue(size_t max_size = 10)
        : max_size_(max_size)
    {
    }
    
    /**
     * @brief デストラクタ
     */
    ~ThreadSafeQueue() = default;
    
    // コピー・ムーブ禁止
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue(ThreadSafeQueue&&) = delete;
    ThreadSafeQueue& operator=(ThreadSafeQueue&&) = delete;
    
    // ========================================================================
    // パブリックメソッド
    // ========================================================================
    
    /**
     * @brief キューに要素を追加（タイムアウトあり）
     * 
     * @param value 追加する要素
     * @param timeout_ms タイムアウト時間（ミリ秒）
     * @return 成功時true、タイムアウト時false
     * 
     * キューが満杯の場合、タイムアウト時間内に空きが出るまで待機します。
     * タイムアウトに達した場合、falseを返します。
     */
    bool push(const T& value, int timeout_ms = 1000) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // キューに空きができるまで待機（タイムアウトあり）
        if (!cond_not_full_.wait_for(lock, 
                std::chrono::milliseconds(timeout_ms),
                [this] { return queue_.size() < max_size_; })) {
            return false;  // タイムアウト
        }
        
        // 要素を追加
        queue_.push(value);
        
        // 空でない条件を通知
        cond_not_empty_.notify_one();
        
        return true;
    }
    
    /**
     * @brief キューから要素を取得（タイムアウトあり）
     * 
     * @param value 取得した要素の格納先
     * @param timeout_ms タイムアウト時間（ミリ秒）
     * @return 成功時true、タイムアウト時false
     * 
     * キューが空の場合、タイムアウト時間内に要素が追加されるまで待機します。
     * タイムアウトに達した場合、falseを返します。
     */
    bool pop(T& value, int timeout_ms = 1000) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // キューに要素が追加されるまで待機（タイムアウトあり）
        if (!cond_not_empty_.wait_for(lock,
                std::chrono::milliseconds(timeout_ms),
                [this] { return !queue_.empty(); })) {
            return false;  // タイムアウト
        }
        
        // 要素を取得
        value = queue_.front();
        queue_.pop();
        
        // 満杯でない条件を通知
        cond_not_full_.notify_one();
        
        return true;
    }
    
    /**
     * @brief キューのサイズを取得
     * 
     * @return 現在のキューのサイズ
     * 
     * この関数はスレッドセーフです。
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
private:
    std::queue<T> queue_;                     ///< 内部キュー
    size_t max_size_;                         ///< 最大サイズ
    mutable std::mutex mutex_;                ///< ミューテックス
    std::condition_variable cond_not_full_;   ///< 満杯でない条件変数
    std::condition_variable cond_not_empty_;  ///< 空でない条件変数
};

} // namespace Integration
} // namespace VirtualAd
