export interface CrunkerConstructorOptions {
    /**
     * Sample rate for Crunker's internal audio context.
     *
     * @default 44100
     */
    sampleRate: number;
}
export type CrunkerInputTypes = string | File | Blob;
/**
 * An exported Crunker audio object.
 */
export interface ExportedCrunkerAudio {
    blob: Blob;
    url: string;
    element: HTMLAudioElement;
}
/**
 * Crunker is the simple way to merge, concatenate, play, export and download audio files using the Web Audio API.
 */
export default class Crunker {
    private readonly _sampleRate;
    private readonly _context;
    /**
     * Creates a new instance of Crunker with the provided options.
     *
     * If `sampleRate` is not defined, it will auto-select an appropriate sample rate
     * for the device being used.
     */
    constructor({ sampleRate }?: Partial<CrunkerConstructorOptions>);
    /**
     * Creates Crunker's internal AudioContext.
     *
     * @internal
     */
    private _createContext;
    /**
     *
     * The internal AudioContext used by Crunker.
     */
    get context(): AudioContext;
    /**
     * Asynchronously fetches multiple audio files and returns an array of AudioBuffers.
     */
    fetchAudio(...filepaths: CrunkerInputTypes[]): Promise<AudioBuffer[]>;
    /**
     * Merges (layers) multiple AudioBuffers into a single AudioBuffer.
     *
     * **Visual representation:**
     *
     * ![](https://user-images.githubusercontent.com/12958674/88806278-968f0680-d186-11ea-9cb5-8ef2606ffcc7.png)
     */
    mergeAudio(buffers: AudioBuffer[]): AudioBuffer;
    /**
     * Concatenates multiple AudioBuffers into a single AudioBuffer.
     *
     * **Visual representation:**
     *
     * ![](https://user-images.githubusercontent.com/12958674/88806297-9d1d7e00-d186-11ea-8cd2-c64cb0324845.png)
     */
    concatAudio(buffers: AudioBuffer[]): AudioBuffer;
    /**
     * Pads a specified AudioBuffer with silence from a specified start time,
     * for a specified length of time.
     *
     * Accepts float values as well as whole integers.
     *
     * @param buffer AudioBuffer to pad
     * @param padStart Time to start padding (in seconds)
     * @param seconds Duration to pad for (in seconds)
     */
    padAudio(buffer: AudioBuffer, padStart?: number, seconds?: number): AudioBuffer;
    /**
     * Slices an AudioBuffer from the specified start time to the end time, with optional fade in and out.
     *
     * @param buffer AudioBuffer to slice
     * @param start Start time (in seconds)
     * @param end End time (in seconds)
     * @param fadeIn Fade in duration (in seconds, default is 0)
     * @param fadeOut Fade out duration (in seconds, default is 0)
     */
    sliceAudio(buffer: AudioBuffer, start: number, end: number, fadeIn?: number, fadeOut?: number): AudioBuffer;
    /**
     * Plays the provided AudioBuffer in an AudioBufferSourceNode.
     */
    play(buffer: AudioBuffer): AudioBufferSourceNode;
    /**
     * Exports the specified AudioBuffer to a Blob, Object URI and HTMLAudioElement.
     *
     * Note that changing the MIME type does not change the actual file format. The
     * file format will **always** be a WAVE file due to how audio is stored in the
     * browser.
     *
     * @param buffer Buffer to export
     * @param type MIME type (default: `audio/wav`)
     */
    export(buffer: AudioBuffer, type?: string): ExportedCrunkerAudio;
    /**
     * Downloads the provided Blob.
     *
     * @param blob Blob to download
     * @param filename An optional file name to use for the download (default: `crunker`)
     */
    download(blob: Blob, filename?: string): HTMLAnchorElement;
    /**
     * Executes a callback if the browser does not support the Web Audio API.
     *
     * Returns the result of the callback, or `undefined` if the Web Audio API is supported.
     *
     * @param callback callback to run if the browser does not support the Web Audio API
     */
    notSupported<T>(callback: () => T): T | undefined;
    /**
     * Closes Crunker's internal AudioContext.
     */
    close(): this;
    /**
     * Returns the largest duration of the longest AudioBuffer.
     *
     * @internal
     */
    private _maxDuration;
    /**
     * Returns the largest number of channels in an array of AudioBuffers.
     *
     * @internal
     */
    private _maxNumberOfChannels;
    /**
     * Returns the sum of the lengths of an array of AudioBuffers.
     *
     * @internal
     */
    private _totalLength;
    /**
     * Returns whether the browser supports the Web Audio API.
     *
     * @internal
     */
    private _isSupported;
    /**
     * Writes the WAV headers for the specified Float32Array.
     *
     * Returns a DataView containing the WAV headers and file content.
     *
     * @internal
     */
    private _writeHeaders;
    /**
     * Converts a Float32Array to 16-bit PCM.
     *
     * @internal
     */
    private _floatTo16BitPCM;
    /**
     * Writes a string to a DataView at the specified offset.
     *
     * @internal
     */
    private _writeString;
    /**
     * Converts an AudioBuffer to a Float32Array.
     *
     * @internal
     */
    private _interleave;
    /**
     * Creates an HTMLAudioElement whose source is the specified Blob.
     *
     * @internal
     */
    private _renderAudioElement;
    /**
     * Creates an Object URL for the specified Blob.
     *
     * @internal
     */
    private _renderURL;
}
