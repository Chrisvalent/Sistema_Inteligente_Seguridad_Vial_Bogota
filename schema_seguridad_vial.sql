--
-- PostgreSQL database dump
--

\restrict zeEf2O3S3MlXbKTrhiNAxwRX2AF7BJDap0cajABhKgdk3aTo8jSZtMF51Ex2dro

-- Dumped from database version 18.6
-- Dumped by pg_dump version 18.6

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: predicciones; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.predicciones (
    id bigint NOT NULL,
    fecha_hora timestamp with time zone DEFAULT now() NOT NULL,
    modelo character varying(100) NOT NULL,
    prediccion character varying(50) NOT NULL,
    umbral_muertos double precision,
    puntuaciones jsonb NOT NULL,
    variables_entrada jsonb NOT NULL
);


--
-- Name: predicciones_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.predicciones_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: predicciones_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.predicciones_id_seq OWNED BY public.predicciones.id;


--
-- Name: predicciones id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.predicciones ALTER COLUMN id SET DEFAULT nextval('public.predicciones_id_seq'::regclass);


--
-- Name: predicciones predicciones_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.predicciones
    ADD CONSTRAINT predicciones_pkey PRIMARY KEY (id);


--
-- PostgreSQL database dump complete
--

\unrestrict zeEf2O3S3MlXbKTrhiNAxwRX2AF7BJDap0cajABhKgdk3aTo8jSZtMF51Ex2dro

