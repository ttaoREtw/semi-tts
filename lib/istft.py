import torch

def istft(
    stft_matrix,  # type: Tensor
    n_fft,  # type: int
    hop_length=None,  # type: Optional[int]
    win_length=None,  # type: Optional[int]
    window=None,  # type: Optional[Tensor]
    center=True,  # type: bool
    pad_mode="reflect",  # type: str
    normalized=False,  # type: bool
    onesided=True,  # type: bool
    length=None,  # type: Optional[int]
):
    # type: (...) -> Tensor
    r"""Inverse short time Fourier Transform. This is expected to be the inverse of torch.stft.
    It has the same parameters (+ additional optional parameter of ``length``) and it should return the
    least squares estimation of the original signal. The algorithm will check using the NOLA condition (
    nonzero overlap).
    Important consideration in the parameters ``window`` and ``center`` so that the envelop
    created by the summation of all the windows is never zero at certain point in time. Specifically,
    :math:`\sum_{t=-\infty}^{\infty} w^2[n-t\times hop\_length] \cancel{=} 0`.
    Since stft discards elements at the end of the signal if they do not fit in a frame, the
    istft may return a shorter signal than the original signal (can occur if ``center`` is False
    since the signal isn't padded).
    If ``center`` is True, then there will be padding e.g. 'constant', 'reflect', etc. Left padding
    can be trimmed off exactly because they can be calculated but right padding cannot be calculated
    without additional information.
    Example: Suppose the last window is:
    [17, 18, 0, 0, 0] vs [18, 0, 0, 0, 0]
    The n_frames, hop_length, win_length are all the same which prevents the calculation of right padding.
    These additional values could be zeros or a reflection of the signal so providing ``length``
    could be useful. If ``length`` is ``None`` then padding will be aggressively removed
    (some loss of signal).
    [1] D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time Fourier transform,"
    IEEE Trans. ASSP, vol.32, no.2, pp.236-243, Apr. 1984.
    Args:
        stft_matrix (torch.Tensor): Output of stft where each row of a channel is a frequency and each
            column is a window. it has a size of either (channel, fft_size, n_frames, 2) or (
            fft_size, n_frames, 2)
        n_fft (int): Size of Fourier transform
        hop_length (Optional[int]): The distance between neighboring sliding window frames.
            (Default: ``win_length // 4``)
        win_length (Optional[int]): The size of window frame and STFT filter. (Default: ``n_fft``)
        window (Optional[torch.Tensor]): The optional window function.
            (Default: ``torch.ones(win_length)``)
        center (bool): Whether ``input`` was padded on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            (Default: ``True``)
        pad_mode (str): Controls the padding method used when ``center`` is True. (Default:
            ``'reflect'``)
        normalized (bool): Whether the STFT was normalized. (Default: ``False``)
        onesided (bool): Whether the STFT is onesided. (Default: ``True``)
        length (Optional[int]): The amount to trim the signal by (i.e. the
            original signal length). (Default: whole signal)
    Returns:
        torch.Tensor: Least squares estimation of the original signal of size
        (channel, signal_length) or (signal_length)
    """
    stft_matrix_dim = stft_matrix.dim()
    assert 4 <= stft_matrix_dim <= 5, "Incorrect stft dimension: %d" % (stft_matrix_dim)

    if stft_matrix_dim == 4:
        # add a channel dimension
        stft_matrix = stft_matrix.unsqueeze(-4)

    dtype = stft_matrix.dtype
    device = stft_matrix.device
    n_channel = stft_matrix.size(-4)
    batch_size = stft_matrix.size(0)
    fft_size = stft_matrix.size(-3)
    assert (onesided and n_fft // 2 + 1 == fft_size) or (
        not onesided and n_fft == fft_size
    ), (
        "one_sided implies that n_fft // 2 + 1 == fft_size and not one_sided implies n_fft == fft_size. "
        + "Given values were onesided: %s, n_fft: %d, fft_size: %d"
        % ("True" if onesided else False, n_fft, fft_size)
    )

    # use stft defaults for Optionals
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    # There must be overlap
    assert 0 < hop_length <= win_length
    assert 0 < win_length <= n_fft

    if window is None:
        window = torch.ones(win_length, requires_grad=False, device=device, dtype=dtype)

    assert window.dim() == 1 and window.size(0) == win_length

    if win_length != n_fft:
        # center window with pad left and right zeros
        left = (n_fft - win_length) // 2
        window = torch.nn.functional.pad(window, (left, n_fft - win_length - left))
        assert window.size(0) == n_fft
    # win_length and n_fft are synonymous from here on

    stft_matrix = stft_matrix.transpose(-3, -2)  # size (batch_size, channel, n_frames, fft_size, 2)
    stft_matrix = torch.irfft(
        stft_matrix, 1, normalized, onesided, signal_sizes=(n_fft,)
    )  # size (batch_size, channel, n_frames, n_fft)

    assert stft_matrix.size(-1) == n_fft
    n_frames = stft_matrix.size(-2)

    ytmp = stft_matrix * window.view(1, 1, 1, n_fft)  # size (batch_size, channel, n_frames, n_fft)
    # each column of a channel is a frame which needs to be overlap added at the right place
    ytmp = ytmp.transpose(-2, -1)  # size (batch_size, channel, n_fft, n_frames)

    eye = torch.eye(n_fft, requires_grad=False, device=device, dtype=dtype).unsqueeze(
        1
    )  # size (n_fft, 1, n_fft)

    # this does overlap add where the frames of ytmp are added such that the i'th frame of
    # ytmp is added starting at i*hop_length in the output
    y = torch.nn.functional.conv_transpose1d(
        ytmp.view(-1, n_fft, n_frames), eye, stride=hop_length, padding=0
    ).view(batch_size, n_channel, 1, -1)  # size (batch_size, channel, 1, expected_signal_len)

    # do the same for the window function
    window_sq = (
        window.pow(2).view(n_fft, 1).repeat((1, n_frames)).unsqueeze(0)
    )  # size (1, n_fft, n_frames)
    window_envelop = torch.nn.functional.conv_transpose1d(
        window_sq, eye, stride=hop_length, padding=0
    )  # size (1, 1, expected_signal_len)

    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    assert y.size(-1) == expected_signal_len
    assert window_envelop.size(2) == expected_signal_len

    half_n_fft = n_fft // 2
    # we need to trim the front padding away if center
    start = half_n_fft if center else 0
    end = -half_n_fft if length is None else start + length

    y = y[..., start:end]
    window_envelop = window_envelop[:, :, start:end]

    # check NOLA non-zero overlap condition
    window_envelop_lowest = window_envelop.abs().min()
    assert window_envelop_lowest > 1e-11, "window overlap add min: %f" % (
        window_envelop_lowest
    )

    y = (y / window_envelop.unsqueeze(0)).squeeze(1)  # size (channel, expected_signal_len)

    if stft_matrix_dim == 4:  # remove the channel dimension
        y = y.squeeze(-2)
    return y
