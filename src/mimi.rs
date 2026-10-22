use crate::{conv, quantization, seanet, transformer as bt, transformer};
use xn::nn::var_builder::Path;
use xn::streaming::{StreamMask, StreamTensor};
use xn::{Backend, Result, Tensor, WithDTypeF};

#[derive(Debug, Copy, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ResampleMethod {
    Conv,
    Interpolate,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub channels: usize,
    pub sample_rate: f64,
    pub frame_rate: f64,
    pub renormalize: bool,
    pub resample_method: ResampleMethod,
    pub seanet: seanet::Config,
    pub transformer: transformer::Config,
    pub quantizer_n_q: usize,
    pub quantizer_bins: usize,
    pub quantizer_dim: usize,
}

impl Config {
    pub fn v0_1(num_codebooks: Option<usize>) -> Self {
        let seanet_cfg = seanet::Config {
            dimension: 512,
            channels: 1,
            causal: true,
            n_filters: 64,
            n_residual_layers: 1,
            activation: seanet::Activation::Elu(1.),
            compress: 2,
            dilation_base: 2,
            disable_norm_outer_blocks: 0,
            final_activation: None,
            kernel_size: 7,
            residual_kernel_size: 3,
            last_kernel_size: 3,
            lstm: None,
            norm: conv::Norm::WeightNorm,
            pad_mode: conv::PadMode::Constant,
            ratios: vec![8, 6, 5, 4],
            true_skip: true,
        };
        let transformer_cfg = transformer::Config {
            d_model: seanet_cfg.dimension,
            num_heads: 8,
            num_layers: 8,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: Some(0.01),
            context: 250,
            use_conv_block: false,
            max_period: 10000.0,
            gating: None,
            norm: crate::NormType::LayerNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            dim_feedforward: 2048,
            kv_repeat: 1,
            head_dim: None,
            final_norm: None,
            proj_bias: false,
            conv_layout: true,
        };
        Config {
            channels: 1,
            sample_rate: 24_000.,
            frame_rate: 12.5,
            renormalize: true,
            resample_method: ResampleMethod::Conv,
            seanet: seanet_cfg,
            transformer: transformer_cfg,
            quantizer_n_q: num_codebooks.unwrap_or(16),
            quantizer_bins: 2048,
            quantizer_dim: 256,
        }
    }

    pub fn v0_1_48khz(num_codebooks: Option<usize>) -> Self {
        let seanet_cfg = seanet::Config {
            dimension: 512,
            channels: 1,
            causal: true,
            n_filters: 64,
            n_residual_layers: 1,
            activation: seanet::Activation::Elu(1.),
            compress: 2,
            dilation_base: 2,
            disable_norm_outer_blocks: 0,
            final_activation: None,
            kernel_size: 7,
            residual_kernel_size: 3,
            last_kernel_size: 3,
            lstm: None,
            norm: conv::Norm::WeightNorm,
            pad_mode: conv::PadMode::Constant,
            ratios: vec![8, 8, 6, 5],
            true_skip: true,
        };
        let transformer_cfg = transformer::Config {
            d_model: seanet_cfg.dimension,
            num_heads: 8,
            num_layers: 8,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: Some(0.01),
            context: 250,
            use_conv_block: false,
            max_period: 10000.0,
            gating: None,
            norm: crate::NormType::LayerNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            dim_feedforward: 2048,
            kv_repeat: 1,
            head_dim: None,
            final_norm: None,
            proj_bias: false,
            conv_layout: true,
        };
        Config {
            channels: 1,
            sample_rate: 48_000.,
            frame_rate: 12.5,
            renormalize: true,
            resample_method: ResampleMethod::Conv,
            seanet: seanet_cfg,
            transformer: transformer_cfg,
            quantizer_n_q: num_codebooks.unwrap_or(16),
            quantizer_bins: 2048,
            quantizer_dim: 256,
        }
    }
}

// ============================================================================
// Streaming State Types
// ============================================================================

pub struct MimiEncodeState<T: WithDTypeF, B: Backend> {
    model: std::sync::Arc<Mimi<T, B>>,
    pub encoder: seanet::EncoderState<T, B>,
    pub encoder_transformer: bt::BatchedTransformerState<T, B>,
    pub downsample: conv::Conv1dState<T, B>,
}

impl<T: WithDTypeF, B: Backend> MimiEncodeState<T, B> {
    pub fn reset_batch_idx(&mut self, batch_idx: usize) -> Result<()> {
        self.encoder.reset_batch_idx(batch_idx)?;
        self.encoder_transformer.reset_batch_idx(batch_idx)?;
        self.downsample.reset_batch_idx(batch_idx)?;
        Ok(())
    }
}

pub struct MimiDecodeState<T: WithDTypeF, B: Backend> {
    model: std::sync::Arc<Mimi<T, B>>,
    pub upsample: conv::ConvTr1dState<T, B>,
    pub decoder_transformer: bt::BatchedTransformerState<T, B>,
    pub decoder: seanet::DecoderState<T, B>,
}

impl<T: WithDTypeF, B: Backend> MimiDecodeState<T, B> {
    pub fn reset_batch_idx(&mut self, batch_idx: usize) -> Result<()> {
        self.upsample.reset_batch_idx(batch_idx)?;
        self.decoder_transformer.reset_batch_idx(batch_idx)?;
        self.decoder.reset_batch_idx(batch_idx)?;
        Ok(())
    }
}

// ============================================================================
// Mimi
// ============================================================================

/// The decoding half of [`Mimi`], absent when it was loaded with
/// [`Mimi::load_encoder`].
struct MimiDecoder<T: WithDTypeF, B: Backend> {
    decoder: seanet::SeaNetDecoder<T, B>,
    decoder_transformer: bt::BatchedProjectedTransformer<xn::Unquantized<T, B>>,
    upsample: conv::ConvTrUpsample1d<T, B>,
}

pub struct Mimi<T: WithDTypeF, B: Backend> {
    encoder: seanet::SeaNetEncoder<T, B>,
    encoder_transformer: bt::BatchedProjectedTransformer<xn::Unquantized<T, B>>,
    downsample: conv::ConvDownsample1d<T, B>,
    quantizer: quantization::SplitResidualVectorQuantizer<T, B>,
    decoder: Option<MimiDecoder<T, B>>,
    config: Config,
}

impl<T: WithDTypeF, B: Backend> Mimi<T, B> {
    /// The encoder and the decoder.
    pub fn load(vb: &Path<B>, cfg: Config) -> Result<Self> {
        let mut mimi = Self::load_encoder(vb, cfg)?;
        let dim = mimi.config.seanet.dimension;
        let decoder = seanet::SeaNetDecoder::load(&vb.pp("decoder"), &mimi.config.seanet)?;
        let decoder_transformer = bt::BatchedProjectedTransformer::load(
            &vb.pp("decoder_transformer"),
            dim,
            &mimi.config.transformer,
        )?;
        let upsample = conv::ConvTrUpsample1d::load(
            &vb.pp("upsample"),
            mimi.downsample_stride(),
            dim,
            /* causal */ true,
            /* learnt */ true,
        )?;
        mimi.decoder = Some(MimiDecoder { decoder, decoder_transformer, upsample });
        Ok(mimi)
    }

    /// Only what encoding needs: `encoder`, `encoder_transformer`,
    /// `downsample` and `quantizer` under `vb`. Speech-to-text never
    /// decodes, and the ASR GGUFs written by the `quantize` example leave
    /// the decoder out. [`decode`](Self::decode) and
    /// [`init_decode_state`](Self::init_decode_state) then return an error.
    pub fn load_encoder(vb: &Path<B>, cfg: Config) -> Result<Self> {
        let dim = cfg.seanet.dimension;

        let encoder = seanet::SeaNetEncoder::load(&vb.pp("encoder"), &cfg.seanet)?;
        let encoder_transformer = bt::BatchedProjectedTransformer::load(
            &vb.pp("encoder_transformer"),
            dim,
            &cfg.transformer,
        )?;

        let quantizer = quantization::SplitResidualVectorQuantizer::load(
            &vb.pp("quantizer"),
            cfg.quantizer_dim,
            Some(dim),
            Some(dim),
            cfg.quantizer_n_q,
            cfg.quantizer_bins,
        )?;

        let downsample = conv::ConvDownsample1d::load(
            &vb.pp("downsample"),
            Self::stride(&cfg),
            dim,
            /* causal */ true,
            /* learnt */ true,
        )?;

        Ok(Self { encoder, encoder_transformer, quantizer, downsample, decoder: None, config: cfg })
    }

    fn stride(cfg: &Config) -> usize {
        let encoder_frame_rate =
            cfg.sample_rate / cfg.seanet.ratios.iter().product::<usize>() as f64;
        (encoder_frame_rate / cfg.frame_rate) as usize
    }

    fn downsample_stride(&self) -> usize {
        Self::stride(&self.config)
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Whether it can decode: loaded with [`load`](Self::load), not
    /// [`load_encoder`](Self::load_encoder).
    pub fn has_decoder(&self) -> bool {
        self.decoder.is_some()
    }

    fn decoder(&self) -> Result<&MimiDecoder<T, B>> {
        self.decoder
            .as_ref()
            .ok_or_else(|| xn::Error::Msg("mimi: loaded without its decoder".into()))
    }

    pub fn init_encode_state(
        self: &std::sync::Arc<Self>,
        batch_size: usize,
    ) -> Result<MimiEncodeState<T, B>> {
        let state = MimiEncodeState {
            model: self.clone(),
            encoder: self.encoder.init_state(),
            encoder_transformer: self.encoder_transformer.init_state(batch_size)?,
            downsample: self.downsample.init_state(),
        };
        Ok(state)
    }

    pub fn init_decode_state(
        self: &std::sync::Arc<Self>,
        batch_size: usize,
    ) -> Result<MimiDecodeState<T, B>> {
        let dec = self.decoder()?;
        let state = MimiDecodeState {
            model: self.clone(),
            upsample: dec.upsample.init_state(),
            decoder_transformer: dec.decoder_transformer.init_state(batch_size)?,
            decoder: dec.decoder.init_state(),
        };
        Ok(state)
    }

    pub fn encode_pre_quantize(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let batch_size = xs.dim(0)?;
        let xs = self.encoder.forward(xs)?;
        let mut tf_state = self.encoder_transformer.init_state(batch_size)?;
        let mask = StreamMask::all_active(batch_size);
        let xs = self.encoder_transformer.forward(&xs, &mut tf_state, &mask)?;
        let xs = &xs[0];
        self.downsample.forward(xs)
    }

    /// Encode audio to codes (non-streaming).
    pub fn encode(&self, xs: &Tensor<T, B>) -> Result<Tensor<i64, B>> {
        let xs = self.encode_pre_quantize(xs)?;
        self.quantizer.encode(&xs)
    }

    /// Decode codes to audio (non-streaming).
    pub fn decode(&self, codes: &Tensor<i64, B>) -> Result<Tensor<T, B>> {
        let dec = self.decoder()?;
        let batch_size = codes.dim(0)?;
        let emb = self.quantizer.decode(codes)?;
        let emb = dec.upsample.forward(&emb)?;
        let mut tf_state = dec.decoder_transformer.init_state(batch_size)?;
        let mask = StreamMask::all_active(batch_size);
        let outs = dec.decoder_transformer.forward(&emb, &mut tf_state, &mask)?;
        dec.decoder.forward(&outs[0])
    }
}

impl<T: WithDTypeF, B: Backend> MimiEncodeState<T, B> {
    /// Encode audio step (streaming).
    pub fn encode_step(
        &mut self,
        xs: &StreamTensor<T, B>,
        mask: &StreamMask,
    ) -> Result<StreamTensor<i64, B>> {
        let model = &self.model;
        let xs = model.encoder.step(xs, &mut self.encoder, mask)?;
        let xs = model.encoder_transformer.step(&xs, &mut self.encoder_transformer, mask)?;
        let xs = model.downsample.step(&xs, &mut self.downsample, mask)?;
        match xs.as_option() {
            None => Ok(StreamTensor::empty()),
            Some(xs) => Ok(StreamTensor::from_tensor(model.quantizer.encode(xs)?)),
        }
    }
}

impl<T: WithDTypeF, B: Backend> MimiDecodeState<T, B> {
    /// Decode codes step (streaming).
    pub fn decode_step(
        &mut self,
        codes: &StreamTensor<i64, B>,
        mask: &StreamMask,
    ) -> Result<StreamTensor<T, B>> {
        let model = &self.model;
        // A decode state is only made from a model with a decoder.
        let dec = model.decoder()?;
        let emb: StreamTensor<T, B> = match codes.as_option() {
            Some(codes) => StreamTensor::from_tensor(model.quantizer.decode(codes)?),
            None => StreamTensor::empty(),
        };
        let emb = dec.upsample.step(&emb, &mut self.upsample, mask)?;
        let out = dec.decoder_transformer.step(&emb, &mut self.decoder_transformer, mask)?;
        dec.decoder.step(&out, &mut self.decoder, mask)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    type Cpu = xn::CpuDevice;

    /// A Mimi small enough to build in a test: 4x downsampling in SEANet,
    /// then 2x, one transformer layer and two codebooks.
    fn tiny_config() -> Config {
        let mut cfg = Config::v0_1(Some(2));
        cfg.sample_rate = 400.;
        cfg.frame_rate = 50.;
        cfg.seanet.dimension = 16;
        cfg.seanet.n_filters = 4;
        cfg.seanet.ratios = vec![2, 2];
        cfg.transformer.d_model = 16;
        cfg.transformer.num_heads = 2;
        cfg.transformer.num_layers = 1;
        cfg.transformer.dim_feedforward = 32;
        cfg.transformer.context = 8;
        cfg.quantizer_bins = 8;
        cfg.quantizer_dim = 8;
        cfg
    }

    /// `tensors` as a safetensors file: the header's length, the header, the
    /// f32 data.
    fn safetensors(tensors: &BTreeMap<String, Vec<usize>>) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape) in tensors {
            let start = data.len();
            let n: usize = shape.iter().product();
            for _ in 0..n {
                data.extend_from_slice(&0.01f32.to_le_bytes());
            }
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [start, data.len()],
                }),
            );
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&header);
        file.extend_from_slice(&data);
        file
    }

    fn load(
        tensors: &BTreeMap<String, Vec<usize>>,
        encoder_only: bool,
        cfg: &Config,
    ) -> Result<Mimi<f32, Cpu>> {
        let vb = xn::nn::VB::from_bytes(vec![safetensors(tensors)], xn::CPU)?;
        let root = vb.root();
        if encoder_only {
            Mimi::load_encoder(&root, cfg.clone())
        } else {
            Mimi::load(&root, cfg.clone())
        }
    }

    /// Every tensor a load asks for, with the shape it expects: added one by
    /// one from its "not found" and "shape mismatch" errors.
    fn weights_for(encoder_only: bool, cfg: &Config) -> BTreeMap<String, Vec<usize>> {
        let mut tensors = BTreeMap::new();
        for _ in 0..10_000 {
            let msg = match load(&tensors, encoder_only, cfg) {
                Ok(_) => return tensors,
                Err(e) => e.to_string(),
            };
            let name = msg.split('\'').nth(1).unwrap_or_else(|| panic!("{msg}")).to_string();
            let shape = match msg.split("expected ").nth(1) {
                Some(rest) => {
                    let dims = rest.split(']').next().unwrap().trim_start_matches('[');
                    dims.split(',').map(|d| d.trim().parse().unwrap()).collect()
                }
                None => vec![1],
            };
            tensors.insert(name, shape);
        }
        panic!("the load never succeeded");
    }

    #[test]
    fn load_encoder_needs_no_decoder_weights() {
        let cfg = tiny_config();
        let tensors = weights_for(true, &cfg);
        assert!(!tensors.is_empty());
        assert!(
            tensors.keys().all(|k| !k.starts_with("decoder") && !k.starts_with("upsample")),
            "{:?}",
            tensors.keys().collect::<Vec<_>>()
        );
        // The full load wants the decoder too.
        assert!(load(&tensors, false, &cfg).is_err());

        let mimi = load(&tensors, true, &cfg).unwrap();
        assert!(!mimi.has_decoder());
        // 64 samples at 4x then 2x: 8 frames of 2 codebooks.
        let pcm = Tensor::<f32, Cpu>::from_vec(vec![0.1; 64], (1, 1, 64), &xn::CPU).unwrap();
        let codes = mimi.encode(&pcm).unwrap();
        assert_eq!(codes.dims(), &[1, 2, 8]);
        let err = mimi.decode(&codes).expect_err("decoding without a decoder");
        assert!(err.to_string().contains("without its decoder"), "{err}");
        let mimi = std::sync::Arc::new(mimi);
        assert!(mimi.init_encode_state(1).is_ok());
        assert!(mimi.init_decode_state(1).is_err());
    }

    #[test]
    fn load_still_loads_the_decoder() {
        let cfg = tiny_config();
        let tensors = weights_for(false, &cfg);
        assert!(tensors.keys().any(|k| k.starts_with("decoder.")));
        assert!(tensors.keys().any(|k| k.starts_with("decoder_transformer.")));
        assert!(tensors.keys().any(|k| k.starts_with("upsample.")));
        let mimi = load(&tensors, false, &cfg).unwrap();
        assert!(mimi.has_decoder());
        let pcm = Tensor::<f32, Cpu>::from_vec(vec![0.1; 64], (1, 1, 64), &xn::CPU).unwrap();
        let codes = mimi.encode(&pcm).unwrap();
        let out = mimi.decode(&codes).unwrap();
        assert_eq!(out.dims(), &[1, 1, 64]);
    }
}
