// #include "sampler.h"

// #include <oneapi/tbb/enumerable_thread_specific.h>
// #include <tbb/enumerable_thread_specific.h>

// #include <boost/thread/sync_bounded_queue.hpp>

// #include "dataset.h"
// #include "stack.h"
// #include "types.h"

// namespace speechpipe {
// sampler_handle sample_dataset(dataset_handle d) {
//     struct sampled_dataset final : sampler {
//         dataset_handle p_base;
//         tbb::enumerable_thread_specific<std::mt19937> rng;
//         explicit sampled_dataset(dataset_handle& p_base) : p_base{p_base} {}
//         item sample() override {
//             bool rng_exists;
//             auto& _rng = rng.local(rng_exists);
//             if (not rng_exists) {
//                 _rng.seed(std::random_device()());
//             }
//             auto dist =
//                 std::uniform_int_distribution<size_t>(0, p_base->size() - 1);
//             auto dice = dist(_rng);

//             auto const& rand_key = p_base->keys[dice];
//             auto it = (*p_base)[rand_key];
//             it.insert_or_assign("key", rand_key);
//             return it;
//         }
//     };
//     return std::make_shared<sampled_dataset>(d);
// }

// sampler_handle sample_samplers(std::vector<sampler_handle> samplers,
//                                std::vector<std::string_view> ids,
//                                std::vector<double> w) {
//     assert(samplers.size() == ids.size() && ids.size() == w.size());

//     struct sampled_samplers final : sampler {
//         std::vector<sampler_handle> p_bases;
//         std::vector<std::string> sampler_ids;
//         std::vector<double> weights;
//         tbb::enumerable_thread_specific<std::mt19937> rng;
//         sampled_samplers(decltype(samplers)&& samplers,
//                          decltype(ids)&& sampler_ids, decltype(w)&& weights)
//             : p_bases{std::move(samplers)},
//               sampler_ids{sampler_ids.begin(), sampler_ids.end()},
//               weights{std::move(weights)} {}
//         item sample() override {
//             bool rng_exist;
//             auto& _rng = rng.local(rng_exist);
//             if (not rng_exist) {
//                 _rng.seed(std::random_device()());
//             }
//             auto dist = std::discrete_distribution<size_t>(weights.begin(),
//                                                            weights.end());
//             auto dice = dist(_rng);
//             auto p_base = p_bases[dice];
//             return p_base->sample();
//         }
//     };
//     return std::make_shared<sampled_samplers>(std::move(samplers),
//                                               std::move(ids), std::move(w));
// }

// sampler_handle map_sampler(sampler_handle s, item_transform func) {
//     struct mapped_sampler final : sampler {
//         sampler_handle base;
//         item_transform func;
//         mapped_sampler(sampler_handle&& base, item_transform&& func)
//             : base{base}, func{std::move(func)} {}
//         item sample() override { return func((*base).sample()); }
//     };
//     return std::make_shared<mapped_sampler>(std::move(s), std::move(func));
// }

// sampler_handle filter_sampler(sampler_handle s, item_pred pred) {
//     struct filtered_sampler final : sampler {
//         sampler_handle base;
//         item_pred pred;
//         filtered_sampler(sampler_handle&& base, item_pred&& pred)
//             : base{base}, pred{std::move(pred)} {}
//         item sample() override {
//             auto item = (*base).sample();
//             while (not pred(item)) {
//                 item = (*base).sample();
//             }
//             return item;
//         }
//     };
//     return std::make_shared<filtered_sampler>(std::move(s), std::move(pred));
// }

// using queue_t = boost::concurrent::sync_bounded_queue<item>;
// inline static void push_queue_forever(std::stop_token st,
//                                       sampler_handle sampler, queue_t& queue)
//                                       {
//     while (not st.stop_requested()) {
//         try {
//             queue.push(sampler->sample());
//         } catch (...) {
//             // Log and ignore the exception.
//         }
//     }
// }

// sampler_handle queue_sampler(sampler_handle sampler, size_t n_threads,
//                              size_t queue_size) {
//     struct queued_sampler final : sampler {
//         sampler_handle base;
//         std::vector<std::jthread> workers;
//         queue_t q;
//         item sample() override { return q.pull(); }
//         queued_sampler(sampler_handle&& sampler, size_t n_threads,
//                        size_t queue_size)
//             : base{std::move(sampler)}, q(queue_size) {
//             for (size_t i = 0; i < n_threads; ++i) {
//                 workers.emplace_back([this](std::stop_token st) {
//                     push_queue_forever(st, this->base, q);
//                 });
//             }
//         }
//         virtual ~queued_sampler() {
//             for (auto& worker : workers) {
//                 worker.request_stop();
//             }
//             while (!q.empty()) {
//                 q.pull();
//             }
//             for (auto& worker : workers) {
//                 if (worker.joinable()) {
//                     worker.join();
//                 }
//             }
//             q.close();
//         }
//     };
//     return std::make_shared<queued_sampler>(std::move(sampler), n_threads,
//                                             queue_size);
// }

// batch_sampler_handle bucket_sampler(sampler_handle s, std::string_view
// sort_key,
//                                     partition p) {
//     struct bucketized_sampler final : batch_sampler {
//         using bucket_t = std::vector<std::vector<item>>;
//         sampler_handle base;
//         std::string sort_key;
//         partition p;
//         tbb::enumerable_thread_specific<bucket_t> buckets;

//         item_list sample() override {
//             bool exists;
//             auto& _buckets = buckets.local(exists);
//             if (not exists) {
//                 _buckets = bucket_t(p.size());
//             }
//             while (true) {
//                 auto it = base->sample();
//                 auto len = std::get<int>(it[sort_key]);
//                 int bin_idx = -1;
//                 for (int i = 0; i < p.size(); ++i) {
//                     auto [a, b, c] = p[i];
//                     if (a <= len and len < b) {
//                         _buckets[i].push_back(std::move(it));
//                         bin_idx = i;
//                         break;
//                     }
//                 }
//                 if (bin_idx == -1) {
//                     continue;  // Drop this item
//                 }
//                 auto [a, b, c] = p[bin_idx];
//                 if (_buckets[bin_idx].size() == c) {
//                     item_list items = std::move(_buckets[bin_idx]);
//                     _buckets[bin_idx].clear();
//                     std::sort(items.begin(), items.end(),
//                               [this](auto&& u, auto&& v) {
//                                   auto u_len = std::get<int>(u[sort_key]);
//                                   auto v_len = std::get<int>(v[sort_key]);
//                                   return v_len < u_len;
//                               });
//                     return items;
//                 }
//             }
//         }

//         bucketized_sampler(sampler_handle s, std::string_view sort_key,
//                            partition p)
//             : base{std::move(s)}, sort_key{sort_key}, p{std::move(p)} {}
//     };
//     return std::make_shared<bucketized_sampler>(s, sort_key, p);
// }

// batch_sampler_handle sample_fixed_batch(sampler_handle s, size_t batch_size)
// {
//     struct fixed_size_batch_sampler final : batch_sampler {
//         sampler_handle base;
//         size_t batch_size;
//         fixed_size_batch_sampler(sampler_handle base, size_t bs)
//             : base{std::move(base)}, batch_size(bs) {}

//         item_list sample() override {
//             item_list items;
//             for (int i = 0; i < batch_size; ++i) {
//                 items.push_back(base->sample());
//             }
//             return items;
//         }
//     };
//     return std::make_shared<fixed_size_batch_sampler>(s, batch_size);
// }

// // TODO: Support other datatypes, only supporting int array now.
// sampler_handle stack_batch(batch_sampler_handle s, key_list int_keys,
//                            key_list int_arr_keys) {
//     struct stacking_sampler final : sampler {
//         batch_sampler_handle base;
//         key_list int_keys;
//         key_list int_arr_keys;
//         stacking_sampler(batch_sampler_handle base, key_list int_keys,
//                          key_list int_arr_keys)
//             : base{std::move(base)},
//               int_keys{std::move(int_keys)},
//               int_arr_keys{std::move(int_arr_keys)} {}
//         item sample() override {
//             auto items = base->sample();
//             return stack_items(items, int_keys, int_arr_keys);
//         }
//     };
//     return std::make_shared<stacking_sampler>(
//         std::move(s), std::move(int_arr_keys), std::move(int_keys));
// }

// sampler_handle sampler::queue(size_t n_threads, size_t queue_size) {
//     return queue_sampler(shared_from_this(), n_threads, queue_size);
// }

// sampler_handle sampler::filter(item_pred pred) {
//     return filter_sampler(shared_from_this(), std::move(pred));
// }

// sampler_handle sampler::map(item_transform func) {
//     return map_sampler(shared_from_this(), std::move(func));
// }

// batch_sampler_handle sampler::batch(size_t batch_size) {
//     return sample_fixed_batch(shared_from_this(), batch_size);
// }

// sampler_handle batch_sampler::stack(key_list int_keys, key_list int_arr_keys)
// {
//     return stack_batch(shared_from_this(), std::move(int_keys),
//                        std::move(int_arr_keys));
// }

// sampler_handle batch_sampler::stack_auto() {
//     auto example_items = sample();
//     auto& example_item = example_items.front();
//     key_list int_keys = keys_of_type<int>(example_item);
//     key_list int_arr_keys = keys_of_type<int_arr>(example_item);
//     return stack(int_keys, int_arr_keys);
// }

// sampler_handle batch_sampler::flatten() {
//     struct flattened_batch_sampler final : sampler {
//         batch_sampler_handle base;
//         key_list int_keys;
//         key_list int_arr_keys;
//         tbb::enumerable_thread_specific<item_list> lists;
//         flattened_batch_sampler(batch_sampler_handle base)
//             : base{std::move(base)} {}
//         item sample() override {
//             item_list& lst = lists.local();
//             while (lst.empty()) {
//                 lst = base->sample();
//             }
//             item it = std::move(lst.back());
//             lst.pop_back();
//             return it;
//         }
//     };
//     return std::make_shared<flattened_batch_sampler>(shared_from_this());
// }

// batch_sampler_handle sampler::bucket(std::string_view sort_key, partition p)
// {
//     return bucket_sampler(shared_from_this(), sort_key, std::move(p));
// }

// sampler_handle sampler::zip(dataset_handle d) {
//     return map([this, d](item it) {
//         std::string const& key = std::get<std::string>(it["key"]);
//         auto dit = (*d)[key];
//         it.merge(std::move(dit));
//         return it;
//     });
// }

// }  // namespace speechpipe