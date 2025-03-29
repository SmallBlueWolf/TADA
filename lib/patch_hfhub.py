import huggingface_hub
import huggingface_hub.file_download

# 模拟旧版 huggingface_hub.cached_download 的实现
def cached_download(
    url,
    library_name=None,
    library_version=None,
    user_agent=None,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    local_files_only=False,
    use_auth_token=None,
    _commit_hash=None,
    legacy_cache_layout=False
):
    """
    这个函数签名根据旧版本 huggingface_hub.cached_download 的原始定义进行参考。
    其中最核心的逻辑是调用新的 _hf_hub_download 或 hf_hub_download 来实现下载及缓存功能。
    """
    # 在 huggingface_hub 0.29.3 以后，官方推荐使用 hf_hub_download。
    # 但因为 diffusers 内部调用的参数仅限旧函数，这里统一映射到新的内部实现。
    return huggingface_hub.file_download._hf_hub_download(
        url,
        library_name=library_name,
        library_version=library_version,
        user_agent=user_agent,
        cache_dir=cache_dir,
        force_download=force_download,
        proxies=proxies,
        resume_download=resume_download,
        local_files_only=local_files_only,
        use_auth_token=use_auth_token,
        _commit_hash=_commit_hash,
        legacy_cache_layout=legacy_cache_layout,
    )

# 通过 setattr 或 monkey patch 为 huggingface_hub 注入旧的 cached_download 接口
setattr(huggingface_hub, "cached_download", cached_download)
