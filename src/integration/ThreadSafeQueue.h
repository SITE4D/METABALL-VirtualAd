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
    // パブリックメソッド（Part 2で実装）
    // ========================================================================
    
    bool push(const T& value, int timeout_ms = 1000);
    bool pop(T& value, int timeout_ms = 1000);
    size_t size() const;
    
private:
    std::queue<T> queue_;                     ///< 内部キュー
    size_t max_size_;                         ///< 最大サイズ
    mutable std::mutex mutex_;                ///< ミューテックス
    std::condition_variable cond_not_full_;   ///< 満杯でない条件変数
    std::condition_variable cond_not_empty_;  ///< 空でない条件変数
};

} // namespace Integration
} // namespace VirtualAd
